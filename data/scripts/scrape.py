import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlsplit, urlunsplit

import click
import requests
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)


def setup_logging(log_file: Path, verbose: bool):
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    level = logging.DEBUG if verbose else logging.INFO
    root_logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)


@dataclass
class Sonnet:
    title: str
    text: str
    original_url: str
    is_valid: bool = True

    @property
    def filename(self) -> str:
        """Generates a filename from the title."""
        name = self.title.strip()
        name = unicodedata.normalize("NFC", name)
        name = name.replace("/", "-").replace("\\", "-").replace(":", " - ")
        name = re.sub(r'[<>:"|?*\x00-\x1F]', "", name)

        filename = f"{name or 'untitled'}.txt"

        if not self.is_valid:
            return f"[CHECK] {filename}"
        return filename


class SonnetCleaner:
    """Utilities for cleaning and formatting sonnet texts."""

    @staticmethod
    def clean_verse(text: str) -> str:
        """Normalizes whitespace and line breaks."""
        text = text.replace("\r\n", " ").replace("\n\r", " ")
        text = re.sub(r"\n\s{4}", " ", text)
        text = re.sub(r"\(\d{1,2}\)", "", text)
        return "\n".join(line.strip() for line in text.split("\n"))

    @staticmethod
    def format_structure(lines: List[str]) -> str:
        """Formats lines into standard (4-4-3-3) or caudato (4-4-3-3-3) layout."""
        count = len(lines)

        if count == 14:
            # Standard Sonnet
            return "\n\n".join(
                [
                    "\n".join(lines[0:4]),
                    "\n".join(lines[4:8]),
                    "\n".join(lines[8:11]),
                    "\n".join(lines[11:14]),
                ]
            )
        elif count == 17:
            # Caudato
            base = SonnetCleaner.format_structure(lines[0:14])
            coda = "\n".join(lines[14:17])
            return f"{base}\n\n{coda}"
        else:
            logger.warning(
                f"Irregular line count ({count}). saving without formatting."
            )
            return "\n".join(lines)

    @staticmethod
    def fix_merged_lines(text: str) -> str:
        """
        Heuristic to fix lines that were accidentally merged.
        Splits lines > 65 chars at punctuation boundaries.
        """
        raw_lines = [line.strip() for line in text.split("\n") if line.strip()]
        fixed_lines = []

        for line in raw_lines:
            if len(line) > 65:
                # Look for punctuation followed by space and capital letter
                match = re.search(r"([:;,.?!])\s+([A-Za-z].*)", line)
                if match:
                    split_index = match.start(1) + 1
                    fixed_lines.append(line[:split_index].strip())
                    fixed_lines.append(line[split_index:].strip())
                    continue
            fixed_lines.append(line)

        return SonnetCleaner.format_structure(fixed_lines)

    @staticmethod
    def validate_structure(text: str) -> bool:
        """Checks if text matches standard sonnet stanza structure."""
        stanzas = re.split(r"\n\s*\n", text.strip())
        counts = [len(s.split("\n")) for s in stanzas if s.strip()]
        return (
            counts == [4, 4, 3, 3]
            or counts == [4, 4, 3, 3, 3]
            or counts == [4, 4, 3, 3, 3, 3]
        )


class Scraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (compatible; belliGPT/1.0; +for-research)"}
        )

    def fetch(self, url: str, sleep: float = 0.5) -> str:
        try:
            r = self.session.get(url, timeout=30)
            r.raise_for_status()
            r.encoding = r.apparent_encoding
            time.sleep(sleep)
            return r.text
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            raise

    @staticmethod
    def normalize_url(url: str) -> str:
        """Clean and format IntraText URLs."""
        parts = urlsplit(url)
        clean = urlunsplit(parts._replace(scheme="https", query="", fragment=""))
        return re.sub(r"/_(\w+).HTM$", r"/__\1.HTM", clean, flags=re.IGNORECASE)

    def get_index_links(self, index_url: str) -> List[str]:
        """Extracts sonnet links from the main index page."""
        logger.info(f"Fetching index: {index_url}")
        html = self.fetch(index_url)
        soup = BeautifulSoup(html, "html.parser")

        links = []
        container = soup.find("div", {"id": "post-body-6731700490332175000"})
        if container:
            for a in container.select("a[href]"):
                href = a["href"]
                if "intratext.com/IXT/ITA1554" in href:
                    links.append(str(href))

        logger.info(f"Found {len(links)} links.")
        return links

    def parse_sonnet_page(self, html: str, url: str) -> Sonnet:
        """Parses a single IntraText sonnet page."""
        soup = BeautifulSoup(html, "html.parser")

        # Extract title
        title = self._extract_title(soup)
        if not title:
            title = f"Unknown_Title_{url}"
            logger.warning(f"Title not found for {url}. Using fallback: {title}")

        # Extract Verses
        verses = []
        paragraphs = soup.find_all(
            "p",
            {
                "style": re.compile(
                    r"margin-left:\d+\.\d+\w{1,}.+text-indent\:\d{1,}\.\d{1,}\w{1,}"
                )
            },
        )

        for p in paragraphs:
            clean_text = self._process_paragraph_text(p)
            if clean_text:
                verses.append(SonnetCleaner.clean_verse(clean_text))

        full_text = "\n\n".join(verses)

        is_valid = SonnetCleaner.validate_structure(full_text)
        if not is_valid:
            fixed_text = SonnetCleaner.fix_merged_lines(full_text)
            if SonnetCleaner.validate_structure(fixed_text):
                full_text = fixed_text
                is_valid = True
                logger.info(f"Fixed structure for: {title}")
            else:
                logger.warning(
                    f"Structure Invalid for: {title}. Saving for manual review."
                )
                is_valid = False

        return Sonnet(title=title, text=full_text, original_url=url, is_valid=is_valid)

    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Find the title in the bs object."""
        # Strategy A: numbered pattern in <p>
        pattern = re.compile(r"^\d+\.\s")
        for p in soup.find_all("p"):
            # Remove superscripts from title candidate
            p_clone = p.__copy__()
            for sup in p_clone.find_all("sup"):
                sup.decompose()
            text = p_clone.get_text(" ", strip=True)
            if pattern.match(text):
                return text.replace("\n", " ").replace("\r", "").strip()

        # Strategy B: list items
        for li in soup.find_all("li"):
            if not li.find("ul"):
                parts = li.get_text(strip=True).split(" . ", 1)
                if len(parts) > 1:
                    return parts[0] + ". " + "".join(parts[1:])
        return None

    def _process_paragraph_text(self, p_tag: Tag) -> str:
        """Extracts and cleans text from a paragraph."""
        p = p_tag.__copy__()

        for br in p.find_all("br"):
            if br.find_parent("sup"):
                parent_sup = br.find_parent("sup")
                br.extract()
                parent_sup.insert_after(br)

        # Remove footnotes
        for sup in p.find_all("sup"):
            sup.decompose()

        # Mark line breaks
        for br in p.find_all("br"):
            br.replace_with(" ||BR|| ")

        raw_text = "".join(p.strings)
        clean_text = " ".join(raw_text.split())
        return clean_text.replace(" ||BR|| ", "\n").replace("||BR||", "")


def save_sonnet(sonnet: Sonnet, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / sonnet.filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(sonnet.text.rstrip() + "\n")


@click.command()
@click.option(
    "--index-url",
    show_default=True,
    help="URL of the sonnet index page.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("data/raw/"),
    show_default=True,
    help="Directory where scraped sonnets are saved.",
)
@click.option(
    "--sleep",
    "sleep_seconds",
    type=float,
    default=0.6,
    show_default=True,
    help="Delay between requests.",
)
@click.option(
    "--limit", type=int, default=None, help="Max number of sonnets to scrape."
)
@click.option(
    "--log-file",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("logs/scrape_sonnets.log"),
    show_default=True,
    help="Path to log file.",
)
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
def main(
    index_url: str,
    output_dir: Path,
    sleep_seconds: float,
    limit: Optional[int],
    log_file: Path,
    verbose: bool,
):
    setup_logging(log_file=log_file, verbose=verbose)

    scraper = Scraper()

    try:
        links = scraper.get_index_links(index_url=index_url)
    except Exception as e:
        logger.critical(f"Could not retrieve index: {e}")
        return

    sonnet_links = links[1:]  # Skip introduction
    if limit is not None:
        sonnet_links = sonnet_links[:limit]

    total = len(sonnet_links)
    logger.info(f"Starting scrape for {total} links.")

    for i, link in enumerate(sonnet_links, start=1):
        clean_url = scraper.normalize_url(link)

        try:
            html = scraper.fetch(clean_url, sleep=sleep_seconds)
            sonnet = scraper.parse_sonnet_page(html, clean_url)

            save_sonnet(sonnet, output_dir)

            status = "OK" if sonnet.is_valid else "REVIEW"
            if i % 10 == 0 or status == "REVIEW":
                logger.info(f"[{i}/{total}] [{status}] {sonnet.title}")

        except Exception as e:
            logger.error(f"[{i}/{total}] Failed processing {clean_url}: {e}")


if __name__ == "__main__":
    main()

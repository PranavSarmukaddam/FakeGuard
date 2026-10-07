"""
download_dataset.py
-------------------
Automatically downloads a public fake news dataset.
Tries WELFake dataset from GitHub (public, no auth needed),
then falls back to the built-in sample dataset.

Run this BEFORE train_model.py:
    python src/download_dataset.py
"""

import os
import sys
import logging
import urllib.request
import zipfile
import io

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# Public dataset sources (no authentication required)
DATASET_SOURCES = [
    {
        "name": "WELFake (Kaggle mirror - GitHub)",
        "fake_url": "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/politifact_fake.csv",
        "real_url": "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/politifact_real.csv",
        "type": "fakenewsnet",
    },
    {
        "name": "ISOT Fake News Dataset (direct CSV)",
        # Hosted on HuggingFace datasets hub (public)
        "fake_url": "https://huggingface.co/datasets/GonzaloA/fake_news/resolve/main/train.csv",
        "type": "huggingface_combined",
    },
]

# Fallback: create a richer built-in dataset
EXTENDED_SAMPLE = {
    "fake": [
        ("SHOCKING: Government secretly controls weather patterns to harm citizens",
         "Sources inside Washington claim officials have been modifying weather for decades using HAARP technology. Leaked documents prove senior officials approved secret weather programs. Scientists who tried to expose the conspiracy were silenced by the Deep State apparatus."),
        ("EXCLUSIVE: Vaccines contain microchips planted by billionaires",
         "Whistleblowers have revealed that millions of vaccine doses contain microscopic tracking devices. The microchips are activated by 5G towers and transmit personal data to a globalist network of billionaires who plan to control the world population."),
        ("Scientists ADMIT: Earth is actually flat and NASA has been lying",
         "A leaked document from NASA confirms the flat-earth theory is real. Multiple insiders have come forward to reveal decades of deception. Satellite images are computer-generated and the International Space Station footage is filmed in a giant underwater pool."),
        ("BREAKING: Moon landing was filmed in Hollywood studio",
         "New footage shows Stanley Kubrick directing the 1969 Apollo mission scenes in a secret film studio. Former CIA operative confirms the entire space race was a propaganda operation. Astronauts were sworn to secrecy under threat of imprisonment."),
        ("EXPOSED: 5G towers spread coronavirus through radio waves",
         "Independent researchers have proved a direct link between 5G tower installations and coronavirus outbreaks. The radiation emitted by 5G networks weakens the immune system making people susceptible to viral infections designed in a laboratory."),
        ("REVEALED: The government is putting fluoride in water to control minds",
         "Internal memos show officials adding mind-control chemicals to tap water supplies across the country. Fluoride at government-approved levels suppresses critical thinking and makes citizens docile and obedient to authority."),
        ("BOMBSHELL: Climate change is a hoax invented by globalists",
         "Recently leaked emails from climate scientists prove that temperature data was fabricated to support the globalist agenda. Carbon taxes are designed to impoverish the middle class and transfer wealth to a small elite group of international bankers."),
        ("SECRET: Chemtrails from planes are biological weapons",
         "Aviation insiders confirm that trails left by commercial planes contain harmful biological agents and heavy metals. The chemtrail program is run by a secret government agency that uses the atmosphere as a delivery mechanism for population control substances."),
        ("ALERT: Eating sugar cures cancer according to suppressed research",
         "Big pharma has hidden studies showing that simple sugar therapy completely defeats malignant tumors in over 90 percent of cases. Oncologists are paid to suppress this cure because cancer treatment is a trillion-dollar industry."),
        ("URGENT: The UN plans to abolish all national governments by 2030",
         "Documents show the UN Agenda 21 and Agenda 2030 plan to end the sovereignty of all nations. A world government controlled by unelected bureaucrats will replace democratic governments. Citizens will be moved into smart cities and monitored by AI surveillance systems."),
        ("CLAIM: Drinking bleach kills COVID-19 inside the body safely",
         "Alternative health gurus recommend diluted household bleach as an effective and safe cure for COVID-19. The mainstream medical establishment has suppressed this information to protect pharmaceutical profits from vaccines and antiviral drugs."),
        ("EXPOSED: Hollywood celebrities are actually lizard people in disguise",
         "Body language experts and former insiders reveal the truth about A-list celebrities. High-resolution camera footage shows reptilian eye slits and scale-like skin patterns confirming that the global elite are a non-human species that has infiltrated positions of power."),
        ("BREAKING: Bill Gates admits depopulation agenda in secret recording",
         "A leaked audio recording from a private Davos meeting features Bill Gates discussing a plan to reduce global population by 15 percent using vaccines and engineered food shortages. The recording was suppressed by mainstream media outlets who receive funding from the Gates Foundation."),
        ("EXCLUSIVE: FDA approves poison disguised as food additive",
         "Internal FDA documents reveal that a common food additive used in processed foods has been known to cause cancer and neurological damage for over 30 years. Food industry lobbyists have paid officials to keep this information out of public awareness."),
        ("REVEALED: Smartphones secretly record all conversations for intelligence agencies",
         "Former NSA contractor confirms that every smartphone regardless of brand records audio continuously and uploads it to intelligence agency servers. The microphone cannot be disabled even when the phone appears to be switched off."),
        ("SHOCK REPORT: Tap water contains mind-altering drug that causes depression",
         "A independent laboratory found traces of an undisclosed pharmaceutical compound in municipal water supplies that has been linked to increased rates of depression and anxiety. Officials refuse to investigate after pressure from pharmaceutical companies."),
    ],
    "real": [
        ("Federal Reserve raises interest rates amid inflation concerns",
         "The Federal Reserve on Wednesday raised its benchmark federal-funds rate by a quarter percentage point as it continues its effort to bring inflation down toward the central bank's 2 percent target. The decision was unanimous among voting members of the Federal Open Market Committee."),
        ("Scientists discover new exoplanet in habitable zone",
         "NASA astronomers announced the detection of a rocky exoplanet orbiting within the habitable zone of a nearby star. The planet, designated Kepler-452b, is approximately 1.6 times the size of Earth and completes one orbit every 385 days."),
        ("Supreme Court rules on landmark environmental case",
         "The Supreme Court issued a major decision on environmental regulation limiting the EPA authority to set broad carbon emission standards for power plants without specific congressional authorization. The ruling was seen as a significant setback for Biden administration climate goals."),
        ("Tech giants report mixed earnings as ad revenue declines",
         "Major technology companies released quarterly earnings reports showing mixed results as digital advertising revenue continued to fall. Meta reported a 4 percent revenue decline while Alphabet posted revenue growth of 6 percent year-over-year."),
        ("WHO updates global health guidelines for respiratory infections",
         "The World Health Organization revised its recommendations for preventing respiratory infections following new evidence from clinical trials. Updated guidance emphasizes improved indoor ventilation and targeted mask use in high-risk healthcare settings."),
        ("European Parliament votes on new data privacy legislation",
         "Members of the European Parliament approved updated digital rights laws affecting how technology companies handle personal data of EU citizens. The legislation introduces stricter requirements for data minimization and user consent."),
        ("Scientists achieve breakthrough in quantum computing research",
         "Researchers at MIT announced progress in error-correcting quantum bits, a key challenge in building practical quantum computers. The team demonstrated a new technique that reduces computational errors by 50 percent compared to previous methods."),
        ("Global food prices rise sharply after supply chain disruptions",
         "International food commodity prices climbed to multi-year highs following disruptions caused by armed conflict in major grain-producing regions and extreme weather events that damaged harvest yields across three continents."),
        ("Electric vehicle sales surge in major markets worldwide",
         "Electric vehicle registrations rose significantly across Europe, China and North America during the second quarter, driven by new model launches and expanded government incentives for charging infrastructure development."),
        ("Public health officials urge caution as flu season begins",
         "Health authorities are reminding the public to get vaccinated as this year's influenza season appears to be beginning earlier than usual. Officials recommend vaccination for all individuals over six months of age as the most effective prevention measure."),
        ("Stock markets recover after weeks of volatility",
         "Global stock markets staged a broad recovery on Friday after several weeks of sharp swings driven by uncertainty over central bank interest rate policy and mixed signals from corporate earnings reports."),
        ("New climate report warns of accelerating sea level rise",
         "A major new study published in Nature Climate Change warns that sea levels could rise faster than previously projected due to accelerating instability in Antarctic and Greenland ice sheets driven by warming ocean temperatures."),
        ("Parliament passes new infrastructure spending bill",
         "Legislators approved a comprehensive infrastructure spending bill that allocates funding for road repairs, bridge maintenance, public transit upgrades and rural broadband expansion over the next decade."),
        ("Medical researchers identify new Alzheimer's risk gene",
         "Researchers have identified a previously unknown genetic variant that significantly increases the risk of late-onset Alzheimer's disease. The discovery could lead to new screening tools and potential therapeutic targets for prevention."),
        ("Central bank holds rates steady amid mixed economic signals",
         "The central bank voted to hold interest rates unchanged at its latest policy meeting as policymakers weighed persistent inflationary pressures against signs of slowing economic growth and a cooling labor market."),
        ("University researchers develop biodegradable plastic alternative",
         "Scientists at a leading university have developed a fully biodegradable plastic alternative derived from agricultural waste that degrades within months under natural conditions, compared to centuries for conventional petroleum-based plastics."),
    ]
}


def _create_extended_csv(out_dir: str):
    """Write extended built-in dataset as Fake.csv and True.csv."""
    import csv
    fake_path = os.path.join(out_dir, "Fake.csv")
    true_path = os.path.join(out_dir, "True.csv")

    with open(fake_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["title", "text", "subject", "date"])
        for title, body in EXTENDED_SAMPLE["fake"]:
            w.writerow([title, body, "Fake News", "2024-01-01"])

    with open(true_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["title", "text", "subject", "date"])
        for title, body in EXTENDED_SAMPLE["real"]:
            w.writerow([title, body, "World News", "2024-01-01"])

    log.info("Created extended sample dataset: %d fake + %d real articles",
             len(EXTENDED_SAMPLE["fake"]), len(EXTENDED_SAMPLE["real"]))
    return True


def _download_fakenewsnet(source: dict, out_dir: str) -> bool:
    """Download FakeNewsNet politifact CSV files."""
    try:
        import csv, io
        fake_path = os.path.join(out_dir, "Fake.csv")
        true_path = os.path.join(out_dir, "True.csv")

        log.info("Downloading fake news CSV from %s ...", source["fake_url"])
        with urllib.request.urlopen(source["fake_url"], timeout=20) as r:
            fake_content = r.read().decode("utf-8")

        log.info("Downloading real news CSV from %s ...", source["real_url"])
        with urllib.request.urlopen(source["real_url"], timeout=20) as r:
            real_content = r.read().decode("utf-8")

        # FakeNewsNet has: id, news_url, title, tweet_ids
        # We need to adapt it - check if it has useful content
        fake_rows = list(csv.DictReader(io.StringIO(fake_content)))
        real_rows = list(csv.DictReader(io.StringIO(real_content)))

        if len(fake_rows) < 5 or len(real_rows) < 5:
            log.warning("Downloaded data too small, skipping this source.")
            return False

        import csv as csv_mod
        with open(fake_path, "w", newline="", encoding="utf-8") as f:
            w = csv_mod.writer(f)
            w.writerow(["title", "text", "subject", "date"])
            for row in fake_rows:
                title = row.get("title", "")
                if title:
                    w.writerow([title, title, "Fake", ""])

        with open(true_path, "w", newline="", encoding="utf-8") as f:
            w = csv_mod.writer(f)
            w.writerow(["title", "text", "subject", "date"])
            for row in real_rows:
                title = row.get("title", "")
                if title:
                    w.writerow([title, title, "Real", ""])

        log.info("Downloaded: %d fake, %d real articles", len(fake_rows), len(real_rows))
        return True

    except Exception as e:
        log.warning("Failed to download from %s: %s", source["name"], e)
        return False


def _download_huggingface(source: dict, out_dir: str) -> bool:
    """Download combined CSV from HuggingFace and split into Fake/True."""
    try:
        import csv, io
        log.info("Downloading from HuggingFace: %s ...", source["fake_url"])
        with urllib.request.urlopen(source["fake_url"], timeout=30) as r:
            content = r.read().decode("utf-8")

        rows = list(csv.DictReader(io.StringIO(content)))
        if len(rows) < 10:
            return False

        # Expected columns: title, text, label (0=real, 1=fake)
        fake_rows = [r for r in rows if str(r.get("label", "")).strip() in ("1", "FAKE", "fake")]
        real_rows = [r for r in rows if str(r.get("label", "")).strip() in ("0", "REAL", "real")]

        if len(fake_rows) < 5 or len(real_rows) < 5:
            log.warning("Could not split into fake/real from this source.")
            return False

        fake_path = os.path.join(out_dir, "Fake.csv")
        true_path = os.path.join(out_dir, "True.csv")

        import csv as csv_mod
        def write_csv(path, rows_list):
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv_mod.writer(f)
                w.writerow(["title", "text", "subject", "date"])
                for row in rows_list:
                    w.writerow([
                        row.get("title", ""),
                        row.get("text", row.get("body", "")),
                        row.get("subject", ""),
                        row.get("date", ""),
                    ])

        write_csv(fake_path, fake_rows)
        write_csv(true_path, real_rows)

        log.info("Downloaded: %d fake, %d real articles", len(fake_rows), len(real_rows))
        return True

    except Exception as e:
        log.warning("HuggingFace download failed: %s", e)
        return False


def download_dataset(force: bool = False) -> bool:
    """
    Main download function. Tries online sources first,
    falls back to extended built-in dataset.

    Returns True if any dataset is now present in data/raw/.
    """
    fake_path = os.path.join(RAW_DIR, "Fake.csv")
    true_path = os.path.join(RAW_DIR, "True.csv")

    if not force and os.path.exists(fake_path) and os.path.exists(true_path):
        log.info("Dataset already present in data/raw/. Use force=True to re-download.")
        return True

    log.info("Attempting to download dataset from public sources...")

    # Try online sources
    for source in DATASET_SOURCES:
        log.info("Trying: %s", source["name"])
        try:
            if source.get("type") == "fakenewsnet":
                success = _download_fakenewsnet(source, RAW_DIR)
            elif source.get("type") == "huggingface_combined":
                success = _download_huggingface(source, RAW_DIR)
            else:
                success = False

            if success:
                log.info("Dataset downloaded successfully from: %s", source["name"])
                return True
        except Exception as e:
            log.warning("Source failed: %s", e)
            continue

    # All online sources failed, use extended built-in
    log.warning(
        "All online download attempts failed (no internet / sources unavailable).\n"
        "Using extended built-in dataset (%d fake + %d real articles).\n"
        "For best results, manually place Fake.csv and True.csv in data/raw/\n"
        "from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset",
        len(EXTENDED_SAMPLE["fake"]),
        len(EXTENDED_SAMPLE["real"])
    )
    return _create_extended_csv(RAW_DIR)


if __name__ == "__main__":
    force = "--force" in sys.argv
    success = download_dataset(force=force)
    if success:
        print("\nDataset ready. Now run: python src/train_model.py")
    else:
        print("\nDataset setup failed.")
        sys.exit(1)

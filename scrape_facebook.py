import os, re, sys, json, time, random, logging, argparse, subprocess
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
from selenium.webdriver.common.by import By
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

CHROME_BINARY = r"C:\Users\chaim\Downloads\chrome-win\chrome.exe"
FB_EMAIL      = os.getenv("FB_EMAIL")
FB_PASSWORD   = os.getenv("FB_PASSWORD")

ANGER_KEYWORDS = [
    "غالي","غالية","مشا","حيدوا","رجعو","رجع","واش كاين","امتا",
    "كونيكسيون","مخدامش","مشكل","ضعيف","بطي","تفو","عيقتو",
    "الله يعطيكم الإفلاس","حرام","سرقة","مزعج","خايب","نبغي نلغي",
    "نبدل","انوي","اتصالات","مزبل","حشومة","سرقو",
]

def _has_anger(text): return any(kw in (text or "") for kw in ANGER_KEYWORDS)
def _random_sleep(lo=1.2, hi=2.8): time.sleep(random.uniform(lo, hi))

# ── detect chrome version ─────────────────────────────────────────────────────
def get_chrome_version(binary_path: str) -> Optional[str]:
    try:
        r = subprocess.run([binary_path, "--version"], capture_output=True, text=True, timeout=10)
        m = re.search(r"(\d+\.\d+\.\d+\.\d+)", r.stdout)
        if m: return m.group(1)
    except Exception: pass
    # fallback: read manifest next to chrome.exe
    for fname in ["manifest.json", "LAST_CHANGE"]:
        p = Path(binary_path).parent / fname
        if p.exists():
            m = re.search(r"(\d+\.\d+\.\d+\.\d+)", p.read_text(errors="ignore"))
            if m: return m.group(1)
    return None

# ── download matching chromedriver from CfT ───────────────────────────────────
def _download_cft_chromedriver(chrome_version: str) -> str:
    import urllib.request, zipfile
    major = chrome_version.split(".")[0]
    url   = "https://googlechromelabs.github.io/chrome-for-testing/known-good-versions-with-downloads.json"
    log.info("Fetching Chrome-for-Testing manifest...")
    with urllib.request.urlopen(url, timeout=20) as r:
        manifest = json.loads(r.read())

    matching = [
        v for v in manifest["versions"]
        if v["version"].startswith(f"{major}.")
        and any(d["platform"] == "win64" for d in v.get("downloads", {}).get("chromedriver", []))
    ]
    if not matching:
        raise RuntimeError(f"No CfT chromedriver for Chrome {major}")

    best    = matching[-1]
    dl_url  = next(d["url"] for d in best["downloads"]["chromedriver"] if d["platform"] == "win64")
    ver_str = best["version"]
    log.info(f"Downloading chromedriver {ver_str}...")

    cache   = Path.home() / ".wdm" / "cft" / ver_str
    cache.mkdir(parents=True, exist_ok=True)
    zip_p   = cache / "cd.zip"
    exe_p   = cache / "chromedriver-win64" / "chromedriver.exe"

    if not exe_p.exists():
        urllib.request.urlretrieve(dl_url, zip_p)
        with zipfile.ZipFile(zip_p) as zf: zf.extractall(cache)
        zip_p.unlink(missing_ok=True)

    log.info(f"ChromeDriver ready: {exe_p}")
    return str(exe_p)

# ── build driver ──────────────────────────────────────────────────────────────
def build_driver(headless: bool = False):
    chrome_version = get_chrome_version(CHROME_BINARY)
    major = chrome_version.split(".")[0] if chrome_version else None
    log.info(f"Chrome version detected: {chrome_version or 'unknown'}")

    # ── Try undetected-chromedriver first ─────────────────────────────────────
    try:
        import undetected_chromedriver as uc
        opts = uc.ChromeOptions()
        opts.binary_location = CHROME_BINARY
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--lang=ar-MA")
        opts.add_argument("--start-maximized")
        if headless: opts.add_argument("--headless=new")
        driver = uc.Chrome(options=opts, use_subprocess=True,
                           version_main=int(major) if major else None)
        log.info("Driver: undetected-chromedriver ✅")
        return driver
    except ImportError:
        log.warning("undetected-chromedriver not found. Run:  pip install undetected-chromedriver")

    # ── Fallback: selenium + CfT-matched chromedriver ─────────────────────────
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service

    opts = Options()
    opts.binary_location = CHROME_BINARY
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--lang=ar-MA")
    opts.add_argument("--start-maximized")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    if headless: opts.add_argument("--headless=new")

    if chrome_version:
        try:
            # Try wdm with pinned version first
            from webdriver_manager.chrome import ChromeDriverManager
            driver_path = ChromeDriverManager(driver_version=chrome_version).install()
            log.info(f"ChromeDriver from wdm (pinned {chrome_version})")
        except Exception as e:
            log.warning(f"wdm pinned failed ({e}) — downloading from CfT...")
            driver_path = _download_cft_chromedriver(chrome_version)
    else:
        from webdriver_manager.chrome import ChromeDriverManager
        driver_path = ChromeDriverManager().install()

    driver = webdriver.Chrome(service=Service(driver_path), options=opts)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"}
    )
    log.info("Driver: Selenium + matched ChromeDriver ✅")
    return driver

# ── login ─────────────────────────────────────────────────────────────────────
def login(driver, email: str, password: str) -> bool:
    

    COOKIE_FILE = "fb_cookies.json"
    driver.get("https://www.facebook.com/")
    _random_sleep(2, 3)

    if os.path.exists(COOKIE_FILE):
        log.info("Trying saved cookies...")
        for c in json.load(open(COOKIE_FILE)):
            try: driver.add_cookie(c)
            except: pass
        driver.refresh(); _random_sleep(3, 4)
        if "login" not in driver.current_url:
            log.info("✅ Logged in via cookies"); return True
        log.info("Cookies expired — fresh login")

    try:
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "email")))
        ef = driver.find_element(By.ID, "email")
        ef.clear()
        for ch in email: ef.send_keys(ch); time.sleep(random.uniform(0.05, 0.15))
        _random_sleep(0.5, 1.0)
        pf = driver.find_element(By.ID, "pass")
        pf.clear()
        for ch in password: pf.send_keys(ch); time.sleep(random.uniform(0.05, 0.15))
        _random_sleep(0.5, 1.0)
        driver.find_element(By.NAME, "login").click()
        _random_sleep(5, 7)

        if "login" in driver.current_url or "checkpoint" in driver.current_url:
            log.warning("⚠️  Checkpoint detected — solve manually in browser.")
            input("Press Enter once fully logged in...")

        json.dump(driver.get_cookies(), open(COOKIE_FILE, "w"))
        log.info("✅ Logged in and cookies saved"); return True
    except Exception as e:
        log.error(f"Login error: {e}")
        input("Complete login manually, then press Enter...")
        json.dump(driver.get_cookies(), open(COOKIE_FILE, "w"))
        return True

# ── scraper class ─────────────────────────────────────────────────────────────
class FacebookCommentScraper:
    def __init__(self, driver): self.driver = driver

    def _scroll(self, n=3):
        for _ in range(n):
            self.driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
            _random_sleep(1.5, 2.5)

    def _dismiss_popups(self):
        from selenium.webdriver.common.by import By
        for xp in ["//div[@aria-label='Close']","//div[@aria-label='إغلاق']",
                   "//button[contains(.,'Allow all')]","//button[contains(.,'Only allow essential')]"]:
            for el in self.driver.find_elements(By.XPATH, xp):
                try: el.click(); _random_sleep(0.3, 0.6)
                except: pass

    def _load_more_comments(self, max_clicks=30):
        from selenium.webdriver.common.by import By
        XPATHS = [
            "//div[@role='button' and contains(.,'View more comments')]",
            "//div[@role='button' and contains(.,'عرض المزيد من التعليقات')]",
            "//div[@role='button' and contains(.,'مشاهدة المزيد')]",
        ]
        clicks = 0
        while clicks < max_clicks:
            clicked = False
            for xp in XPATHS:
                for btn in self.driver.find_elements(By.XPATH, xp):
                    try:
                        self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
                        _random_sleep(0.3, 0.6)
                        self.driver.execute_script("arguments[0].click();", btn)
                        _random_sleep(1.5, 2.5); clicked = True; clicks += 1
                    except: pass
            if not clicked: break
        log.info(f"  Load-more clicks: {clicks}")

    def _click_see_more(self):
        from selenium.webdriver.common.by import By
        for xp in ["//div[@role='button' and contains(.,'عرض المزيد')]",
                   "//div[@role='button' and contains(.,'See more')]"]:
            for btn in self.driver.find_elements(By.XPATH, xp):
                try: self.driver.execute_script("arguments[0].click();", btn); _random_sleep(0.3, 0.6)
                except: pass

    def _extract_comments(self, post_url: str) -> List[Dict]:
        from selenium.webdriver.common.by import By
        seen, out = set(), []
        for xp in [
            "//div[@aria-label='Comment']//div[@dir='auto']",
            "//div[contains(@class,'x1y1aw1k')]//div[@dir='auto']",
            "//ul//li//div[@dir='auto']",
            "//div[@role='article']//div[@dir='auto']",
        ]:
            for el in self.driver.find_elements(By.XPATH, xp):
                try:
                    t = el.text.strip()
                    if not t or t in seen or len(t) < 4: continue
                    if any(s in t for s in ["Like","Reply","تعليق","إعجاب","أعجبني"]): continue
                    seen.add(t)
                    out.append({"post_url": post_url, "text": t,
                                "is_anger": _has_anger(t),
                                "scraped_at": datetime.utcnow().isoformat()})
                except: pass
        return out

    def scrape_post(self, url: str, max_clicks=20) -> List[Dict]:
        log.info(f"→ {url}")
        self.driver.get(url); _random_sleep(3, 5)
        self._dismiss_popups()
        self._scroll(2)
        self._load_more_comments(max_clicks)
        self._click_see_more()
        self._scroll(2)
        comments = self._extract_comments(url)
        log.info(f"  {len(comments)} comments extracted")
        return comments

    def _collect_post_urls(self, limit: int) -> List[str]:
        
        urls, scrolls = set(), 0
        while len(urls) < limit and scrolls < limit * 3:
            try:
                anchors = self.driver.find_elements(
                    By.XPATH,
                    "//a[contains(@href,'/posts/') or contains(@href,'/reel/')]"
                )
                for a in anchors:
                    try:
                        href = a.get_attribute("href") or ""
                        if "facebook.com" in href and ("/posts/" in href or "/reel/" in href):
                            clean = href.split("?")[0]
                            urls.add(clean)
                            if len(urls) >= limit:
                                break
                    except StaleElementReferenceException:
                        continue  # element removed from DOM, skip it
            except StaleElementReferenceException:
                pass  # entire list invalidated, will re-query on next loop
            
            if len(urls) >= limit:
                break
                
            self.driver.execute_script("window.scrollBy(0,800);")
            _random_sleep(1.5, 2.5)
            scrolls += 1
            
        return list(urls)[:limit]

    def scrape_page(self, page: str, max_posts=20, max_clicks=50) -> List[Dict]:
        self.driver.get(f"https://www.facebook.com/{page}"); _random_sleep(3, 5)
        self._dismiss_popups()
        urls = self._collect_post_urls(max_posts)
        log.info(f"Collected {len(urls)} post URLs")
        all_c = []
        for i, url in enumerate(urls, 1):
            log.info(f"[{i}/{len(urls)}]")
            try: all_c.extend(self.scrape_post(url, max(5, max_clicks // 10)))
            except Exception as e: log.warning(f"  Skipped: {e}")
            _random_sleep(2, 4)
        return all_c

# ── save / summary ────────────────────────────────────────────────────────────
def save_results(comments, out_dir="data"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv = f"{out_dir}/fb_comments_{ts}.csv"
    jsn = f"{out_dir}/fb_comments_{ts}.json"
    df  = pd.DataFrame(comments)
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    json.dump(comments, open(jsn, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log.info(f"CSV  → {csv}  ({len(df)} rows)")
    log.info(f"JSON → {jsn}")

def print_summary(comments):
    if not comments: print("No comments."); return
    df = pd.DataFrame(comments)
    a  = df["is_anger"].sum()
    print(f"\n{'='*50}\nSCRAPING SUMMARY\n  Total : {len(df)}\n  Anger : {a} ({a/len(df)*100:.1f}%)\n  Posts : {df['post_url'].nunique()}\n{'='*50}")
    for t in df[df["is_anger"]]["text"].head(5): print(f"  • {t[:120]}")

# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--page",     default="orangemaroc")
    p.add_argument("--posts",    type=int, default=20)
    p.add_argument("--comments", type=int, default=50)
    p.add_argument("--url",      help="Single post URL")
    p.add_argument("--email",    default=FB_EMAIL)
    p.add_argument("--password", default=FB_PASSWORD)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--out",      default="data")
    p.add_argument("--no-login", action="store_true")
    args = p.parse_args()

    email, password = args.email, args.password
    if not args.no_login and (not email or not password):
        print("Set FB_EMAIL / FB_PASSWORD in .env  or pass --email / --password")
        email    = input("Email: ").strip()
        password = input("Password: ").strip()

    driver = build_driver(args.headless)
    try:
        if not args.no_login:
            if not login(driver, email, password): sys.exit(1)
        scraper  = FacebookCommentScraper(driver)
        comments = (scraper.scrape_post(args.url, args.comments)
                    if args.url else
                    scraper.scrape_page(args.page, args.posts, args.comments))
        if comments: save_results(comments, args.out); print_summary(comments)
        else: log.warning("No comments collected.")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
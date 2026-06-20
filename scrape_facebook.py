from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException, NoSuchElementException
import subprocess
import time
import re
import json
import hashlib
from datetime import datetime

# ── CONFIG ─────────────────────────────────────────────────────────────
PAGE_NAME = "orangemaroc"
MAX_POSTS = 50
CHROME_BINARY = r"C:\Users\chaim\Downloads\chrome-win\chrome.exe"
MAX_RETRIES = 2

# Blocked usernames — Facebook UI elements that get mistaken for comments
BLOCKED_USERNAMES = {
    "Follow", "Like", "Reply", "Share", "Comment", "Orange", "Meta AI",
    "Groups", "Find friends", "Home", "Create", "Menu", "Notifications",
    "Saved", "Memories", "Privacy", "Terms", "Advertising", "Ad choices",
    "Cookies", "Sweet Pie", "Friends", "Reels", "Feeds", "Events",
    "Ads Manager", "Play games", "For you", "Profile", "Watch",
    "Marketplace", "Messenger", "Search", "Pages", "Settings",
    "Help", "Log Out", "Create Post", "Live", "Gaming", "Fundraisers",
    "Feed", "Marketplace", "Notifications", "Create", "Pages",
}

# ── SETUP ────────────────────────────────────────────────────────────────
options = Options()
options.binary_location = CHROME_BINARY
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--start-maximized")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option("useAutomationExtension", False)

service = Service()
if hasattr(subprocess, 'CREATE_NO_WINDOW'):
    service.creation_flags = subprocess.CREATE_NO_WINDOW

driver = webdriver.Chrome(service=service, options=options)

driver.execute_cdp_cmd(
    "Page.addScriptToEvaluateOnNewDocument",
    {"source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"}
)

# ── LOGIN ────────────────────────────────────────────────────────────────
driver.get("https://www.facebook.com/login")
print("=" * 50)
print("LOG IN MANUALLY in the browser window.")
print("Once you're fully logged in, press Enter here.")
print("=" * 50)
input("Press Enter after login...")

# ── COLLECT POST URLS ────────────────────────────────────────────────────
def collect_post_urls(page_name, limit):
    driver.get(f"https://www.facebook.com/{page_name}")
    print(f"\nLoading page: {page_name}")
    time.sleep(5)

    urls = set()
    scrolls = 0
    max_scrolls = limit * 5

    while len(urls) < limit and scrolls < max_scrolls:
        try:
            anchors = driver.find_elements(
                By.XPATH,
                "//a[contains(@href,'/posts/') or contains(@href,'/reel/')]"
            )
            for a in anchors:
                try:
                    href = a.get_attribute("href") or ""
                    if "facebook.com" in href and ("/posts/" in href or "/reel/" in href):
                        clean = href.split("?")[0]
                        urls.add(clean)
                except StaleElementReferenceException:
                    continue
        except StaleElementReferenceException:
            pass

        if len(urls) >= limit:
            break

        driver.execute_script("window.scrollBy(0, 800);")
        time.sleep(2)
        scrolls += 1

    result = list(urls)[:limit]
    print(f"Collected {len(result)} unique post URLs")
    return result

# ── CLICK COMMENT BUTTON (for reels) ─────────────────────────────────────
def click_comment_button():
    strategies = [
        "//div[@aria-label='Comment' and @role='button']",
        "//div[@aria-label='تعليق' and @role='button']",
        "//div[contains(@class,'x1i10hfl') and @aria-label='Comment']",
    ]
    for xpath in strategies:
        try:
            btn = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, xpath))
            )
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
            time.sleep(0.5)
            driver.execute_script("arguments[0].click();", btn)
            return True
        except TimeoutException:
            continue
        except Exception:
            continue
    return False

# ── CLICK "VIEW MORE COMMENTS" (for posts) ─────────────────────────────
def load_more_comments(max_clicks=20):
    clicks = 0
    while clicks < max_clicks:
        clicked = False
        xpaths = [
            "//div[@role='button' and contains(.,'View more comments')]",
            "//div[@role='button' and contains(.,'عرض المزيد من التعليقات')]",
            "//div[@role='button' and contains(.,'مشاهدة المزيد')]",
            "//span[contains(text(),'View more comments')]/ancestor::div[@role='button']",
        ]
        for xp in xpaths:
            for btn in driver.find_elements(By.XPATH, xp):
                try:
                    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
                    time.sleep(0.5)
                    driver.execute_script("arguments[0].click();", btn)
                    time.sleep(2)
                    clicked = True
                    clicks += 1
                except: pass
        if not clicked:
            break
    return clicks

# ── EXTRACT COMMENTS ─────────────────────────────────────────────────────
def extract_comments(post_url):
    time.sleep(3)
    comments = []
    seen = set()

    try:
        mount = driver.find_element(By.CSS_SELECTOR, "div[id^='mount_0_0_']")
    except:
        return comments

    user_links = mount.find_elements(
        By.XPATH,
        ".//a[contains(@href,'/')][@role='link' or not(@role)]"
    )

    for link in user_links:
        try:
            username = link.text.strip()
            if not username or len(username) < 2:
                continue

            # Skip blocked UI usernames
            if username in BLOCKED_USERNAMES:
                continue

            # Skip short/timestamp-like usernames
            if len(username) < 3:
                continue
            if re.match(r'^\d+[dhwmy]$', username):
                continue
            if re.match(r'^\d+$', username):
                continue

            # Skip URLs and UI paths
            if "facebook.com" in username.lower() or "privacy" in username.lower():
                continue
            if "/" in username:
                continue

            # Walk up to find comment container
            container = link
            comment_text = ""
            for _ in range(6):
                try:
                    container = container.find_element(By.XPATH, "..")
                    container_text = container.text.strip()

                    if username in container_text and len(container_text) > len(username) + 10:
                        lines = container_text.split('\n')

                        comment_lines = []
                        found_username = False
                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                            if line == username or username in line:
                                found_username = True
                                continue
                            if found_username and line not in ["Like", "Reply", "أعجبني", "رد", "·", "Follow", ""]:
                                if re.match(r'^\d+[dhwmy]$', line):
                                    continue
                                if re.match(r'^\d+$', line):
                                    continue
                                if "replied" in line.lower() or "replies" in line.lower():
                                    continue
                                if "See translation" in line and len(line) < 20:
                                    continue
                                comment_lines.append(line)

                        if comment_lines:
                            comment_text = ' '.join(comment_lines)
                            break

                except:
                    break

            # Skip empty or too short comments
            if not comment_text or len(comment_text.strip()) < 3:
                continue

            # Clean text
            comment_text = re.sub(r'\s*See translation\s*', ' ', comment_text)
            comment_text = re.sub(r'\s*Hide translation\s*', ' ', comment_text)
            comment_text = re.sub(r'\s*See more\s*', ' ', comment_text)
            comment_text = re.sub(r'\s*Edited\s*', ' ', comment_text)
            comment_text = ' '.join(comment_text.split())

            customer_id = hashlib.md5(username.encode('utf-8')).hexdigest()[:12]

            dedup_key = f"{customer_id}:{comment_text[:50]}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            comments.append({
                "customer_id": customer_id,
                "post_url": post_url,
                "username": username,
                "text": comment_text,
                "scraped_at": datetime.utcnow().isoformat()
            })

        except:
            continue

    return comments

# ── MAIN ─────────────────────────────────────────────────────────────────
all_comments = []
skipped_posts = []

post_urls = collect_post_urls(PAGE_NAME, MAX_POSTS)

for i, url in enumerate(post_urls, 1):
    print(f"\n{'='*50}")
    print(f"[{i}/{len(post_urls)}] Processing: {url}")
    print(f"{'='*50}")

    is_reel = "/reel/" in url
    is_post = "/posts/" in url

    try:
        driver.get(url)
        time.sleep(4)

        # REELS: click comment button
        if is_reel:
            clicked = False
            for attempt in range(1, MAX_RETRIES + 1):
                if click_comment_button():
                    print("Comment button clicked.")
                    clicked = True
                    break
                else:
                    print(f"Attempt {attempt}/{MAX_RETRIES}: Comment button not found")
                    if attempt < MAX_RETRIES:
                        time.sleep(2)
                    else:
                        print(f"SKIPPING reel after {MAX_RETRIES} failed attempts")
                        skipped_posts.append(url)
            if not clicked:
                continue

        # POSTS: load more comments
        elif is_post:
            print("Post detected — loading more comments...")
            clicks = load_more_comments(max_clicks=15)
            print(f"Clicked 'load more' {clicks} times")

        # Extract
        comments = extract_comments(url)
        all_comments.extend(comments)
        print(f"Extracted {len(comments)} comments")

    except Exception as e:
        print(f"ERROR processing post: {e}")
        continue

    time.sleep(2)

# ── SAVE RESULTS ─────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"SCRAPING COMPLETE")
print(f"Total posts processed: {len(post_urls) - len(skipped_posts)}")
print(f"Skipped posts: {len(skipped_posts)}")
print(f"Total comments: {len(all_comments)}")
print(f"{'='*50}")

if skipped_posts:
    print("\nSkipped URLs:")
    for u in skipped_posts:
        print(f"  • {u}")

if all_comments:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/comments_{ts}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(all_comments, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to: {filename}")

    print("\nSample comments:")
    for c in all_comments[:5]:
        print(f"\n  customer_id: {c['customer_id']}")
        print(f"  username: {c['username']}")
        print(f"  text: {c['text'][:100]}")

input("\nPress Enter to close browser...")
driver.quit()
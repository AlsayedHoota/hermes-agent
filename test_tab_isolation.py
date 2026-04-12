#!/usr/bin/env python3
"""Test CDP tab isolation in browser_tool.py.

Creates two concurrent CDP sessions, navigates each to different URLs,
verifies they don't interfere, and cleans up.
"""
import sys
import os
import json
import threading
import time
import requests

# Add the hermes-agent dir to sys.path so we can import the tool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set the CDP override so the tool uses our Chrome
os.environ["BROWSER_CDP_URL"] = "http://localhost:9222"

from tools.browser_tool import (
    _get_session_info,
    _run_browser_command,
    cleanup_browser,
    _active_sessions,
    _cdp_resolve_tab_index,
)

CDP_HOST = "localhost:9222"

def get_tabs():
    """Get current page tabs from Chrome DevTools API."""
    resp = requests.get(f"http://{CDP_HOST}/json/list", timeout=5)
    return [(i, t.get("id",""), t.get("title", "?"), t.get("url", "?"))
            for i, t in enumerate(resp.json()) if t.get("type") == "page"]

def count_page_tabs():
    resp = requests.get(f"http://{CDP_HOST}/json/list", timeout=5)
    return len([t for t in resp.json() if t.get("type") == "page"])

def find_tab_by_id(target_id):
    for i, tid, title, url in get_tabs():
        if tid == target_id:
            return (i, title, url)
    return None

passed = 0
failed = 0

def check(desc, condition):
    global passed, failed
    if condition:
        print(f"  PASS: {desc}")
        passed += 1
    else:
        print(f"  FAIL: {desc}")
        failed += 1

print("=" * 60)
print("CDP Tab Isolation Test (target_id based)")
print("=" * 60)

initial_tab_count = count_page_tabs()
print(f"\nInitial page tabs: {initial_tab_count}")
for i, tid, title, url in get_tabs():
    print(f"  [{i}] {title[:40]} | {url[:50]}")

# --- Create two sessions ---
print("\n--- Creating session A ---")
session_a = _get_session_info("test_a")
tab_id_a = session_a.get("cdp_tab_id")
print(f"  tab_id={tab_id_a}")

print("--- Creating session B ---")
session_b = _get_session_info("test_b")
tab_id_b = session_b.get("cdp_tab_id")
print(f"  tab_id={tab_id_b}")

new_count = count_page_tabs()
print(f"\nTabs after creation: {new_count}")
check("Two new tabs created", new_count == initial_tab_count + 2)
check("Tab IDs are distinct", tab_id_a != tab_id_b)
check("Tab A exists in /json/list", find_tab_by_id(tab_id_a) is not None)
check("Tab B exists in /json/list", find_tab_by_id(tab_id_b) is not None)

# --- Navigate each to different URL ---
print("\n--- Navigating A -> httpbin.org/get ---")
r = _run_browser_command("test_a", "open", ["https://httpbin.org/get"], timeout=30)
print(f"  success={r.get('success', r.get('data',{}).get('url','?'))}")

print("--- Navigating B -> example.com ---")
r = _run_browser_command("test_b", "open", ["https://example.com"], timeout=30)
print(f"  success={r.get('success', r.get('data',{}).get('url','?'))}")

time.sleep(2)

info_a = find_tab_by_id(tab_id_a)
info_b = find_tab_by_id(tab_id_b)
print(f"\n  Tab A: {info_a}")
print(f"  Tab B: {info_b}")
check("Tab A is on httpbin.org/get", info_a and "httpbin.org/get" in info_a[2])
check("Tab B is on example.com", info_b and "example.com" in info_b[2])

# --- Concurrent navigation ---
print("\n--- Concurrent navigation test ---")
errors = []

def nav_a():
    try:
        r = _run_browser_command("test_a", "open", ["https://httpbin.org/html"], timeout=30)
    except Exception as e:
        errors.append(f"A: {e}")

def nav_b():
    try:
        r = _run_browser_command("test_b", "open", ["https://httpbin.org/robots.txt"], timeout=30)
    except Exception as e:
        errors.append(f"B: {e}")

t1 = threading.Thread(target=nav_a)
t2 = threading.Thread(target=nav_b)
t1.start(); t2.start()
t1.join(45); t2.join(45)

time.sleep(2)
info_a = find_tab_by_id(tab_id_a)
info_b = find_tab_by_id(tab_id_b)
print(f"  Tab A: {info_a}")
print(f"  Tab B: {info_b}")
check("No errors during concurrent nav", len(errors) == 0)
check("Tab A on /html after concurrent nav", info_a and "httpbin.org/html" in info_a[2])
check("Tab B on /robots.txt after concurrent nav", info_b and "httpbin.org/robots.txt" in info_b[2])

# --- Cleanup ---
print("\n--- Cleanup ---")
pre_cleanup = count_page_tabs()
cleanup_browser("test_a")
time.sleep(1)
cleanup_browser("test_b")
time.sleep(1)
post_cleanup = count_page_tabs()
print(f"  Before cleanup: {pre_cleanup} tabs")
print(f"  After cleanup:  {post_cleanup} tabs")
check("Tabs cleaned up (back to initial count)", post_cleanup == initial_tab_count)
check("Tab A gone from /json/list", find_tab_by_id(tab_id_a) is None)
check("Tab B gone from /json/list", find_tab_by_id(tab_id_b) is None)

# --- Summary ---
print(f"\n{'=' * 60}")
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
print("=" * 60)
sys.exit(0 if failed == 0 else 1)

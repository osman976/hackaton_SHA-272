import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

def yandex_url(address):
    # Адрес: замена спецсимволов для поиска
    addr_str = quote_plus(address.replace(",", ", ").replace("  ", " "))
    return f"https://yandex.ru/maps/?text={addr_str}"

def check_osint(address):
    url = yandex_url(address)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Referer": "https://yandex.ru/maps/",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "ru,en;q=0.9",
        "Connection": "keep-alive"
    }

    try:
        resp = requests.get(
            url,
            headers=headers,
            timeout=10,
            allow_redirects=True  # если вдруг редирект на капчу/антибот
        )
        # DEBUG: покажем, что реально вернулось
        debug_content = resp.text[:1000]
        print(f"==== Проверка адреса: {address}\nURL: {url}\nКод: {resp.status_code}\n---\n{debug_content}\n---\n")

        # Если пришёл не 200, явно ошибка
        if resp.status_code != 200:
            return False, [], [], []

        # Если Яндекс вернул капчу/антибот — чаще всего там есть строка "antirobot"
        if "antirobot" in resp.text or "Похоже, вы робот" in resp.text:
            print("=== ВНИМАНИЕ: Яндекс.Карты требуют антибот-проверку. ===")
            return False, [], [], []

        soup = BeautifulSoup(resp.text, "html.parser")
        found = soup.find(class_="toponym-businesses-list-snippet")
        if found:
            # По адресу найдена коммерческая организация
            return True, ["коммерция"], ["yandex_maps"], [url]
        else:
            return False, [], [], []

    except Exception as ex:
        print(f"[osint_checker] Ошибка при проверке адреса '{address}': {ex}")
        return False, [], [], []

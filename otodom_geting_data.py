import requests
from bs4 import BeautifulSoup
import csv
import lxml.html
import re

regions = ['dolnoslaskie', 'kujawsko-pomorskie', 'lodzkie', 'lubuskie', 'malopolskie',
           'mazowieckie', 'opolskie', 'podkarpackie', 'podlaskie', 'pomorskie', 'slaskie',
           'swietokrzyskie', 'warminsko-mazurskie', 'wielkopolskie', 'zachodniopomorskie', 'lubelskie']

# open csv file
out_file = open('otodom_data.csv', 'w', newline='', encoding='utf-8')
out_csv = csv.writer(out_file)
out_csv.writerow(['rooms', 'area', 'city', 'private_offer', 'price', 'added_rent', 'year', 'features'])

# go through all the regions
for region in regions:

    url_region = f'https://www.otodom.pl/wynajem/mieszkanie/{region}'

    print(url_region)

    # find number of pages for given region
    response = requests.get(url_region)
    tree = lxml.html.fromstring(response.text)
    comment = tree.xpath('//*[@id="pagerForm"]/ul/comment()')[0]
    start_pages = str(comment).find('placeholder') + len('placeholder') + 2
    pages = str(comment)[start_pages:].split('"')[0]

    print(pages)

    # go through all the pages of specified region
    for page in range(1, int(pages) + 1):
        url = f'{url_region}/?page={page}'
        res = requests.get(url)
        soup = BeautifulSoup(res.text, "html.parser")

        # print(soup)

        # find all instances of articles
        links_tags = soup.select("article")

        # go through each article to find all parameters
        for tag in links_tags:
            offer_url = [a['href'] for a in tag.find_all('a', {'data-tracking': "click_body"}) if a.text][0]

            rooms = tag.find("li", {"class": "offer-item-rooms"}).text
            rooms = re.search('[0-9]', rooms).group(0)
            rooms = int(rooms)

            price = tag.find("li", {"class": "offer-item-price"}).\
                text.replace('zł', '').replace(' ', '').replace('/mc', '').replace('\n', '')

            area = tag.find("li", {"class": "offer-item-area"}).text.replace(' m²', '').replace(',', '.').replace(' ',
                                                                                                                  '')
            area = float(area)

            city = tag.find("p", {"class": "text-nowrap"}).text.strip().split(':')[1].split(',')[:2]

            dev = [el.find("li", {"class": "pull-right"}).text.strip() for el in
                   tag.find_all("div", {"class": "offer-item-details-bottom"})]
            private_offer = 1 if dev[0] == 'Oferta prywatna' else 0

            # open offer page and get more data:

            response = requests.get(offer_url)
            offer_soup = BeautifulSoup(response.text, "html.parser")

            print(region, page, offer_url)

            rent = offer_soup.find("div", {"aria-label": "Czynsz - dodatkowo"})
            added_rent = [int(re.findall(r'\d+', rent.text)[0]) if bool(rent) else 0]

            year = offer_soup.find("div", {"aria-label": "Rok budowy"})
            year = [int(re.findall(r'\d+', year.text)[0]) if bool(year) else None]

            features = [el.text.strip() for el in
                        offer_soup.find_all("li", {"data-cy": "ad.ad-features.categorized-list.item-with-category"})]

            # save to CSV
            out_csv.writerow([rooms, area, city[0].strip(), private_offer, price, added_rent, year, features])

out_file.close()

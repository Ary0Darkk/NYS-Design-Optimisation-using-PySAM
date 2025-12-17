# 🍞 Quick Commerce Bread Price Tracker

A Python-based web scraper and dashboard that compares bread prices across India's top quick commerce platforms: **Blinkit**, **Zepto**, and **Swiggy Instamart**.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Selenium](https://img.shields.io/badge/Selenium-4.0+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)

## 📋 Features

- **Multi-Platform Scraping**: Automatically scrapes bread prices from 3 major platforms
- **Smart Product Matching**: Uses fuzzy logic to match identical products across platforms
- **Interactive Dashboard**: Streamlit-based UI with filters and visualizations
- **Price Comparison**: Identifies best deals and potential savings
- **Brand Analytics**: Filter and compare by brands (Harvest Gold, Britannia, Modern, etc.)
- **Real-time Data**: Get current pricing information (when scraping is enabled)

## 🎯 Use Cases

- **Consumers**: Find the cheapest bread prices before ordering
- **Market Research**: Analyze pricing strategies across platforms
- **Price Monitoring**: Track price changes over time
- **Data Analysis**: Study e-commerce pricing patterns

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Google Chrome browser
- ChromeDriver (matching your Chrome version)

### Step 1: Clone or Download

```bash
# If using git
git clone https://github.com/yourusername/bread-price-scraper.git
cd bread-price-scraper

# Or download and extract the ZIP file
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
selenium>=4.0.0
pandas>=1.3.0
streamlit>=1.20.0
fuzzywuzzy>=0.18.0
python-Levenshtein>=0.12.0
```

### Step 3: Install ChromeDriver

**Option A: Automatic (Recommended)**
```bash
pip install webdriver-manager
```

Then update the code to use:
```python
from webdriver_manager.chrome import ChromeDriverManager
driver = webdriver.Chrome(ChromeDriverManager().install(), options=self.options)
```

**Option B: Manual**
1. Check your Chrome version: `chrome://version/`
2. Download matching ChromeDriver from [here](https://chromedriver.chromium.org/downloads)
3. Add to PATH or place in project directory

## 📖 Usage

### Basic Usage

1. **Run the application:**
```bash
streamlit run bread_scraper.py
```

2. **The dashboard will open in your browser** at `http://localhost:8501`

3. **Enable scraping** (see below)

### Enabling Live Scraping

By default, the app runs with **sample data**. To enable live scraping:

1. Open `bread_scraper.py`
2. Find the `if __name__ == "__main__":` section (bottom of file)
3. **Uncomment** these lines:
```python
# scraper.scrape_blinkit()
# scraper.scrape_zepto()
# scraper.scrape_instamart()
```

To:
```python
scraper.scrape_blinkit()
scraper.scrape_zepto()
scraper.scrape_instamart()
```

4. Save and restart the app

### Running Individual Scrapers

```python
from bread_scraper import BreadScraper

scraper = BreadScraper()

# Scrape only one platform
scraper.scrape_blinkit()

# Access the data
import pandas as pd
df = pd.DataFrame(scraper.data)
print(df)
```

## 🔧 Configuration

### Updating Selectors

Websites frequently change their HTML structure. If scraping stops working:

1. **Open the target website** (e.g., blinkit.com)
2. **Search for "bread"**
3. **Right-click on a product** → **Inspect**
4. **Find the class names** in Chrome DevTools
5. **Update selectors in the code:**

```python
# In scrape_blinkit() method
product_selectors = [
    "div[class*='YourNewClassName']",  # Add new selector
    "div[class*='Product']",           # Keep old as fallback
]
```

### Selector Locations:

| Platform | Method | Line Range |
|----------|--------|------------|
| Blinkit | `scrape_blinkit()` | ~60-120 |
| Zepto | `scrape_zepto()` | ~130-170 |
| Instamart | `scrape_instamart()` | ~180-220 |

### Customizing Scraping Parameters

```python
# Change number of products to scrape
for item in products[:50]:  # Default is 30, change to 50

# Adjust wait times
time.sleep(5)  # Increase if pages load slowly

# Modify search URLs
scraper.scrape_blinkit("https://blinkit.com/s/?q=whole%20wheat%20bread")
```

## 📊 Dashboard Features

### Filters
- **Brand Selection**: Filter by specific bread brands
- **Platform Selection**: Compare specific platforms

### Metrics
- Total products scraped
- Number of platforms
- Average price
- Brand count

### Tables
- **Price Comparison**: Side-by-side pricing across platforms
- **Best Deals**: Top 5 products with highest savings
- **Raw Data**: Complete dataset with all fields

## ⚠️ Important Warnings

### Legal Considerations

1. **Terms of Service**: Check each platform's ToS before scraping
2. **robots.txt**: Respect the robots.txt file of each website
3. **Rate Limiting**: Don't overwhelm servers with requests
4. **Personal Use**: This tool is for educational/personal use only

### Technical Limitations

- **Anti-bot measures**: Modern sites may block automated scraping
- **Selector changes**: CSS classes change frequently, requiring updates
- **Dynamic content**: Some elements load via JavaScript with delays
- **IP blocking**: Excessive requests may result in temporary IP bans

### Best Practices

```python
# Add delays between requests
time.sleep(random.uniform(3, 7))

# Use rotating user agents
user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)...",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)..."
]

# Respect rate limits
# Don't run scraper more than once per hour
```

## 🐛 Troubleshooting

### Common Issues

**1. "ChromeDriver not found"**
```bash
# Solution: Install webdriver-manager
pip install webdriver-manager
```

**2. "No products scraped"**
- Selectors may be outdated → Update using inspect method
- Website may be blocking → Try without headless mode
- Network issues → Check internet connection

**3. "ImportError: No module named 'fuzzywuzzy'"**
```bash
pip install fuzzywuzzy python-Levenshtein
```

**4. Scraping is very slow**
- Reduce number of products: `products[:10]`
- Use headless mode (already enabled)
- Check your internet speed

**5. Getting empty prices or names**
```python
# Enable debug mode - add print statements
print(f"Found {len(products)} products")
print(f"Name: {name}, Price: {price_text}")
```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Add to scraper methods
print(f"Attempting to scrape {platform_name}")
print(f"Products found: {len(products)}")
```

## 📁 Project Structure

```
bread-price-scraper/
│
├── bread_scraper.py          # Main application file
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── data/                     # (Optional) Store scraped data
│   └── bread_prices.csv
│
└── logs/                     # (Optional) Store logs
    └── scraper.log
```

## 🔄 Advanced Features

### Save Data to CSV

```python
# After scraping
df = pd.DataFrame(scraper.data)
df.to_csv('bread_prices.csv', index=False)

# Load later
df = pd.read_csv('bread_prices.csv')
```

### Schedule Regular Scraping

Use `cron` (Linux/Mac) or Task Scheduler (Windows):

```bash
# Run every 6 hours
0 */6 * * * cd /path/to/project && python bread_scraper.py
```

### Export Dashboard as Report

```python
# Add to dashboard
if st.button("Export to Excel"):
    pivot_df.to_excel('price_comparison.xlsx')
    st.success("Exported successfully!")
```

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Improvement

- [ ] Add more platforms (BigBasket, Amazon Fresh)
- [ ] Implement proxy rotation
- [ ] Add price history tracking
- [ ] Create mobile app version
- [ ] Add notification for price drops
- [ ] Support for more product categories

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚖️ Disclaimer

This tool is provided for **educational purposes only**. Users are responsible for:
- Complying with website Terms of Service
- Respecting rate limits and robots.txt
- Using data ethically and legally

The authors are not responsible for misuse or any legal consequences.

## 📧 Contact

For questions, issues, or suggestions:
- Open an issue on GitHub
- Email: your.email@example.com
- Twitter: @yourhandle

## 🙏 Acknowledgments

- **Selenium** - Browser automation
- **Streamlit** - Dashboard framework
- **FuzzyWuzzy** - String matching
- **Pandas** - Data manipulation

---

**Happy Price Hunting! 🛒💰**

*Last Updated: December 2025*
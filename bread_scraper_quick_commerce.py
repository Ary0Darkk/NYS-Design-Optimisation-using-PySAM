import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from fuzzywuzzy import fuzz, process
import streamlit as st
import re

class BreadScraper:
    def __init__(self):
        self.options = Options()
        self.options.add_argument("--headless")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument("--disable-blink-features=AutomationControlled")
        self.options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        self.data = []

    def extract_price(self, price_text):
        """Extract numeric price from text"""
        try:
            # Remove currency symbols and extract number
            price_match = re.search(r'(\d+(?:\.\d+)?)', price_text.replace(',', ''))
            if price_match:
                return float(price_match.group(1))
        except:
            pass
        return None

    def extract_weight(self, text):
        """Extract weight/quantity from text"""
        # Look for patterns like 400g, 400 g, 400gm, 1kg, etc.
        weight_match = re.search(r'(\d+(?:\.\d+)?)\s*(g|gm|kg|ml|l|pcs?|pieces?)', text.lower())
        if weight_match:
            return weight_match.group(0)
        return "N/A"

    def extract_brand(self, product_name):
        """Extract brand from product name"""
        common_brands = ['Harvest Gold', 'Britannia', 'Modern', 'English Oven', 
                        'Fresho', 'Bonn', 'Aadhoos', 'Naturefresh', 'Kitty']
        
        for brand in common_brands:
            if brand.lower() in product_name.lower():
                return brand
        
        # If no known brand, take first word
        words = product_name.split()
        return words[0] if words else "Unknown"

    def scrape_blinkit(self, url="https://blinkit.com/s/?q=bread"):
        """Scrape Blinkit with updated selectors"""
        driver = webdriver.Chrome(options=self.options)
        platform_name = "Blinkit"
        
        try:
            driver.get(url)
            wait = WebDriverWait(driver, 10)
            time.sleep(3)  # Allow initial page load
            
            # Updated selectors (may need adjustment based on current site structure)
            # Common patterns: product-* or Product*, plp-product, etc.
            product_selectors = [
                "div[class*='Product']",
                "div[class*='product']",
                ".plp-product",
                "div[class*='ProductCard']"
            ]
            
            products = []
            for selector in product_selectors:
                try:
                    products = driver.find_elements(By.CSS_SELECTOR, selector)
                    if len(products) > 0:
                        break
                except:
                    continue
            
            count = 0
            for item in products[:30]:
                if count >= 30:
                    break
                    
                try:
                    # Try multiple selector patterns for name
                    name = None
                    name_selectors = ["div[class*='name']", "div[class*='Name']", 
                                     "div[class*='title']", "h3", "h4"]
                    for sel in name_selectors:
                        try:
                            name = item.find_element(By.CSS_SELECTOR, sel).text
                            if name:
                                break
                        except:
                            continue
                    
                    if not name:
                        continue
                    
                    # Try multiple selector patterns for price
                    price_text = None
                    price_selectors = ["div[class*='price']", "span[class*='price']",
                                      "div[class*='Price']", "span[class*='Price']"]
                    for sel in price_selectors:
                        try:
                            price_text = item.find_element(By.CSS_SELECTOR, sel).text
                            if price_text:
                                break
                        except:
                            continue
                    
                    if not price_text:
                        continue
                        
                    price = self.extract_price(price_text)
                    if not price:
                        continue
                    
                    # Extract weight from name or separate field
                    weight = self.extract_weight(name)
                    brand = self.extract_brand(name)
                    
                    self.data.append({
                        'Platform': platform_name,
                        'Product Name': name,
                        'Brand': brand,
                        'Weight': weight,
                        'Price': price
                    })
                    count += 1
                    
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"Error scraping {platform_name}: {str(e)}")
        finally:
            driver.quit()

    def scrape_zepto(self, url="https://www.zepto.com/search?query=bread"):
        """Scrape Zepto with updated selectors"""
        driver = webdriver.Chrome(options=self.options)
        platform_name = "Zepto"
        
        try:
            driver.get(url)
            time.sleep(4)
            
            # Zepto-specific selectors
            product_selectors = [
                "div[class*='product']",
                "div[class*='Product']",
                "div[class*='card']"
            ]
            
            products = []
            for selector in product_selectors:
                try:
                    products = driver.find_elements(By.CSS_SELECTOR, selector)
                    if len(products) > 5:
                        break
                except:
                    continue
            
            count = 0
            for item in products[:30]:
                if count >= 30:
                    break
                    
                try:
                    name = item.find_element(By.CSS_SELECTOR, "h4, h3, div[class*='name'], span[class*='name']").text
                    price_text = item.find_element(By.CSS_SELECTOR, "span[class*='price'], div[class*='price']").text
                    
                    price = self.extract_price(price_text)
                    if not price or not name:
                        continue
                    
                    weight = self.extract_weight(name)
                    brand = self.extract_brand(name)
                    
                    self.data.append({
                        'Platform': platform_name,
                        'Product Name': name,
                        'Brand': brand,
                        'Weight': weight,
                        'Price': price
                    })
                    count += 1
                    
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"Error scraping {platform_name}: {str(e)}")
        finally:
            driver.quit()

    def scrape_instamart(self, url="https://www.swiggy.com/instamart/search?custom_back=true&query=bread"):
        """Scrape Swiggy Instamart with updated selectors"""
        driver = webdriver.Chrome(options=self.options)
        platform_name = "Instamart"
        
        try:
            driver.get(url)
            time.sleep(4)
            
            # Instamart-specific selectors
            product_selectors = [
                "div[class*='productCard']",
                "div[class*='ProductCard']",
                "a[class*='product']"
            ]
            
            products = []
            for selector in product_selectors:
                try:
                    products = driver.find_elements(By.CSS_SELECTOR, selector)
                    if len(products) > 5:
                        break
                except:
                    continue
            
            count = 0
            for item in products[:30]:
                if count >= 30:
                    break
                    
                try:
                    name = item.find_element(By.CSS_SELECTOR, "div[class*='name'], div[class*='title']").text
                    price_text = item.find_element(By.CSS_SELECTOR, "span[class*='price'], div[class*='price']").text
                    
                    price = self.extract_price(price_text)
                    if not price or not name:
                        continue
                    
                    weight = self.extract_weight(name)
                    brand = self.extract_brand(name)
                    
                    self.data.append({
                        'Platform': platform_name,
                        'Product Name': name,
                        'Brand': brand,
                        'Weight': weight,
                        'Price': price
                    })
                    count += 1
                    
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"Error scraping {platform_name}: {str(e)}")
        finally:
            driver.quit()

    def match_products(self, df):
        """Match similar products across platforms using fuzzy matching"""
        if df.empty:
            return df
        
        # Create standardized name by removing extra spaces and converting to lowercase
        df['Clean Name'] = df['Product Name'].str.lower().str.strip()
        
        # Group similar products
        df['Standardized Name'] = df['Product Name']
        processed = set()
        
        for idx, row in df.iterrows():
            if row['Clean Name'] in processed:
                continue
                
            # Find similar products across platforms
            similar = df[df['Clean Name'].apply(lambda x: fuzz.ratio(x, row['Clean Name']) > 80)]
            
            if len(similar) > 1:
                # Use the shortest name as standard
                standard_name = similar.loc[similar['Product Name'].str.len().idxmin(), 'Product Name']
                df.loc[similar.index, 'Standardized Name'] = standard_name
                processed.update(similar['Clean Name'].tolist())
        
        return df

# --- Streamlit Dashboard ---
def run_dashboard(df):
    st.set_page_config(page_title="Bread Price Tracker", page_icon="🍞", layout="wide")
    
    st.title("🍞 Quick Commerce Bread Price Tracker")
    st.markdown("Compare bread prices across Blinkit, Zepto, and Instamart")
    
    if df.empty:
        st.error("No data available. Please run the scraper first.")
        return
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    all_brands = sorted(df['Brand'].unique())
    selected_brands = st.sidebar.multiselect(
        "Select Brands", 
        all_brands,
        default=all_brands[:5] if len(all_brands) >= 5 else all_brands
    )
    
    selected_platforms = st.sidebar.multiselect(
        "Select Platforms",
        df['Platform'].unique(),
        default=df['Platform'].unique()
    )
    
    # Filter data
    filtered_df = df[
        (df['Brand'].isin(selected_brands) if selected_brands else True) &
        (df['Platform'].isin(selected_platforms) if selected_platforms else True)
    ]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Products", len(filtered_df))
    with col2:
        st.metric("Platforms", len(selected_platforms))
    with col3:
        st.metric("Avg Price", f"₹{filtered_df['Price'].mean():.2f}")
    with col4:
        st.metric("Brands", len(selected_brands) if selected_brands else len(all_brands))
    
    # Price comparison table
    st.subheader("📊 Price Comparison")
    
    if not filtered_df.empty:
        pivot_df = filtered_df.pivot_table(
            index=['Standardized Name', 'Brand', 'Weight'],
            columns='Platform',
            values='Price',
            aggfunc='min'
        ).reset_index()
        
        # Calculate best price
        platform_cols = [col for col in pivot_df.columns if col in df['Platform'].unique()]
        pivot_df['Best Price'] = pivot_df[platform_cols].min(axis=1)
        pivot_df['Cheapest At'] = pivot_df[platform_cols].idxmin(axis=1)
        pivot_df['Savings'] = pivot_df[platform_cols].max(axis=1) - pivot_df['Best Price']
        
        # Sort by savings
        pivot_df = pivot_df.sort_values('Savings', ascending=False)
        
        st.dataframe(
            pivot_df.style.format({'Best Price': '₹{:.2f}', 'Savings': '₹{:.2f}'}),
            use_container_width=True
        )
        
        # Best deals
        st.subheader("💰 Top 5 Best Deals")
        top_deals = pivot_df.nlargest(5, 'Savings')[
            ['Standardized Name', 'Brand', 'Best Price', 'Cheapest At', 'Savings']
        ]
        st.dataframe(top_deals, use_container_width=True)
        
    else:
        st.warning("No products match your filter criteria.")
    
    # Raw data
    with st.expander("📋 View Raw Data"):
        st.dataframe(filtered_df, use_container_width=True)

if __name__ == "__main__":
    # Initialize scraper
    scraper = BreadScraper()
    
    # Run scrapers (uncomment when ready to use)
    st.info("⚠️ Note: Scraping is currently disabled. Uncomment scraper calls in main block to enable.")
    
    # Uncomment these lines to actually scrape:
    scraper.scrape_blinkit()
    scraper.scrape_zepto()
    scraper.scrape_instamart()
    
    # For demo purposes, create sample data
    sample_data = [
        {'Platform': 'Blinkit', 'Product Name': 'Harvest Gold Bread 400g', 'Brand': 'Harvest Gold', 'Weight': '400g', 'Price': 42},
        {'Platform': 'Zepto', 'Product Name': 'Harvest Gold Bread 400g', 'Brand': 'Harvest Gold', 'Weight': '400g', 'Price': 45},
        {'Platform': 'Instamart', 'Product Name': 'Harvest Gold Bread 400g', 'Brand': 'Harvest Gold', 'Weight': '400g', 'Price': 40},
        {'Platform': 'Blinkit', 'Product Name': 'Britannia Bread 400g', 'Brand': 'Britannia', 'Weight': '400g', 'Price': 38},
        {'Platform': 'Zepto', 'Product Name': 'Britannia Bread 400g', 'Brand': 'Britannia', 'Weight': '400g', 'Price': 37},
        {'Platform': 'Blinkit', 'Product Name': 'Modern Bread 450g', 'Brand': 'Modern', 'Weight': '450g', 'Price': 50},
        {'Platform': 'Instamart', 'Product Name': 'Modern Bread 450g', 'Brand': 'Modern', 'Weight': '450g', 'Price': 48},
    ]
    
    scraper.data = sample_data
    
    # Create DataFrame and match products
    df = pd.DataFrame(scraper.data)
    if not df.empty:
        df = scraper.match_products(df)
        run_dashboard(df)
    else:
        st.error("No data scraped. Please check the scraper configuration.")
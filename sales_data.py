import pandas as pd
import numpy as np
from datetime import datetime
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define constants
NUM_ROWS = 10000
NUM_DISTRIBUTORS = 50
NUM_SKUS = 200
START_YEAR = 2020
END_YEAR = 2025
QUARTERS = [1, 2, 3, 4]

# Define categories and industries
CATEGORIES = ['Electronics', 'Apparel', 'Food', 'Beverages', 'Home Goods', 'Beauty', 'Health', 
              'Toys', 'Sports', 'Automotive', 'Office Supplies', 'Pet Supplies', 'Paint and ancillaries']

INDUSTRIES = ['Retail', 'E-commerce', 'Wholesale', 'Manufacturing', 'Hospitality', 
              'Healthcare', 'Education', 'Financial Services']

# Create movement categories for initial population
MOVEMENT_CATEGORIES = ['Fast Moving', 'Slow Moving', 'Medium']

# Define realistic product names by category
PRODUCT_NAMES = {
    'Electronics': [
        'Daikin 1.5 Ton Inverter AC', 'LG 1 Ton Split AC', 'Voltas 2 Ton Window AC', 
        'Samsung 43" Crystal 4K TV', 'Sony Bravia 55" OLED TV', 'LG 32" HD Ready LED TV',
        'iPhone 15 Pro 256GB', 'Samsung Galaxy S24 Ultra', 'OnePlus 12 5G', 'Google Pixel 8 Pro',
        'Dell XPS 15 Laptop', 'MacBook Air M3', 'HP Pavilion Gaming Laptop', 'Lenovo ThinkPad X1',
        'Bose QuietComfort Earbuds', 'Sony WH-1000XM5 Headphones', 'JBL Flip 6 Speaker',
        'Canon EOS R6 Camera', 'DJI Mavic 3 Drone', 'Philips Air Purifier 3000i'
    ],
    'Apparel': [
        'Levi\'s 501 Original Jeans', 'Nike Air Max Shoes', 'Adidas Ultraboost Sneakers',
        'H&M Basic T-Shirt Pack', 'Zara Slim Fit Blazer', 'Uniqlo AIRism Undershirt',
        'Puma Running Shorts', 'Under Armour Performance Polo', 'Tommy Hilfiger Classic Shirt',
        'Gap Logo Hoodie', 'Calvin Klein Boxer Brief Set', 'Ray-Ban Wayfarer Sunglasses',
        'Fossil Chronograph Watch', 'Louis Vuitton Neverfull Bag', 'Michael Kors Crossbody Bag',
        'Vans Old Skool Sneakers', 'Columbia Fleece Jacket', 'North Face Puffer Jacket'
    ],
    'Food': [
        'Amul Butter 500g', 'Britannia Good Day Cookies', 'MTR Breakfast Mix', 'Nestle Maggi Noodles',
        'MDH Garam Masala', 'Tata Salt 1kg', 'Daawat Basmati Rice 5kg', 'Patanjali Honey 500g',
        'Kellogg\'s Corn Flakes', 'Bournvita 1kg', 'Kissan Mixed Fruit Jam', 'Haldiram\'s Bhujia',
        'Parle-G Biscuits Value Pack', 'Cadbury Dairy Milk Chocolate', 'Lay\'s Classic Chips',
        'Fortune Sunflower Oil 5L', 'Saffola Active Oil 5L', 'Everest Chicken Masala'
    ],
    'Beverages': [
        'Coca-Cola 2L Bottle', 'Pepsi 330ml Can Pack', 'Red Bull Energy Drink', 'Thums Up 1.25L',
        'Amul Gold Milk 1L', 'Tropicana Orange Juice', 'Real Mixed Fruit Juice', 'Bisleri 20L Can',
        'Kinley Water 1L Pack', 'Tata Tea Premium 1kg', 'Nescafe Classic Coffee 200g', 'Horlicks Classic 1kg',
        'Paper Boat Aam Panna', 'Sprite 2.25L Bottle', 'Mountain Dew 750ml', 'Minute Maid Pulpy Orange',
        'Maaza Mango Drink 1.2L', '7UP 600ml Bottle'
    ],
    'Home Goods': [
        'Cello Finesse Dinner Set', 'Prestige Popular Pressure Cooker', 'Milton Thermosteel Flask',
        'Bombay Dyeing Bedsheet Set', 'Spaces Cotton Towel Set', 'Wonderchef Gas Stove',
        'Pigeon Favorite Electric Kettle', 'Philips Dry Iron', 'Bajaj Mixer Grinder',
        'Borosil Glass Set', 'Hawkins Pressure Cooker', 'Tupperware Storage Container Set',
        'Faber Chimney Hood', 'IFB Microwave Oven', 'Morphy Richards Toaster',
        'Supreme Plastic Chair Set', 'Ajanta Wall Clock', 'Havells Table Fan'
    ],
    'Beauty': [
        'Maybelline Fit Me Foundation', 'Lakme Absolute Lipstick', 'MAC Ruby Woo Lipstick',
        'L\'Oreal Paris Shampoo', 'Dove Moisturizing Cream', 'Neutrogena Sunscreen SPF 50',
        'Forest Essentials Face Wash', 'Biotique Morning Nectar Lotion', 'Nivea Men Face Wash',
        'Revlon ColorStay Eyeliner', 'Pantene Anti-Dandruff Shampoo', 'Garnier Color Naturals Hair Color',
        'Olay Total Effects Day Cream', 'Vaseline Body Lotion', 'Himalaya Purifying Neem Face Wash',
        'Head & Shoulders Shampoo', 'Ponds Face Powder', 'VLCC Sunscreen Gel'
    ],
    'Health': [
        'Dettol Hand Sanitizer', 'Savlon Antiseptic Liquid', 'Himalaya Wellness Tablets',
        'Dabur Chyawanprash', 'Patanjali Ashwagandha', 'Baidyanath Chyawanprash',
        'Revital H Capsules', 'Zandu Balm', 'Volini Spray', 'Omron Blood Pressure Monitor',
        'Dr. Morepen Glucometer', 'AccuCheck Sugar Test Strips', 'Pediasure Growth Supplement',
        'Ensure Protein Powder', 'Horlicks Women\'s Plus', 'Supradyn Multivitamin Tablets',
        'Cetaphil Gentle Skin Cleanser', 'Vicks VapoRub'
    ],
    'Toys': [
        'Lego Classic Creative Box', 'Funskool Monopoly', 'Mattel Barbie Dreamhouse',
        'Nerf Elite Blaster', 'Hot Wheels 5-Car Pack', 'Fisher-Price Learning Toys',
        'Hasbro Gaming Jenga', 'Play-Doh Modeling Compound', 'Nintendo Switch Console',
        'PlayStation 5 Digital Edition', 'Xbox Series X', 'UNO Card Game',
        'Rubik\'s Cube Original', 'Chess Set Wooden', 'Carrom Board Full Size',
        'Remote Control Car', 'Disney Princess Doll', 'Marvel Action Figure Set'
    ],
    'Sports': [
        'MRF Cricket Bat', 'SG Cricket Ball Set', 'Yonex Badminton Racket',
        'Nike Football', 'Adidas Running Shoes', 'Cosco Volleyball',
        'Nivia Basketball', 'Fitbit Charge 5', 'Garmin Forerunner Watch',
        'Strauss Yoga Mat', 'Reebok Dumbbells Set', 'Puma Sports Bag',
        'Wilson Tennis Racket', 'Fastrack Sports Watch', 'Slazenger Tennis Balls',
        'Callaway Golf Club Set', 'Speedo Swimming Goggles', 'Columbia Hiking Backpack'
    ],
    'Automotive': [
        'Castrol Engine Oil 3L', 'Shell Helix Ultra 4L', 'Bosch Wiper Blades', 
        'Michelin P215/65R16 Tires', 'Exide Battery 65Ah', 'Amaron Battery 45Ah',
        'JK Tyre 185/65 R15', 'CEAT 175/65 R14 Tyre', 'MRF ZLX 155/80 R13 Tyre',
        'Mobil Super Mileage 15W-40', 'Gulf Pride 4T Plus 10W-30', 'Turtle Wax Polish',
        '3M Car Care Kit', 'Bosch Spark Plug Set', 'Philips Headlight Bulbs',
        'AutoStark Car Cover', 'Honda Engine Oil 10W-30', 'Yamaha Bike Oil 10W-40'
    ],
    'Office Supplies': [
        'Parker Vector Fountain Pen', 'Reynolds Trimax Pen Pack', 'Classmate Notebook Set',
        'HP Laser Printer Paper', 'Staedtler Pencil Set', 'Casio FX-991ES Calculator',
        'Kangaro Stapler HS-10D', 'Camlin Permanent Marker Set', 'Oddy Copier Paper A4',
        'JK Easy Copier Paper', 'Fevicol Adhesive 200g', 'Post-it Notes Value Pack',
        'Scotch Magic Tape', 'Faber-Castell Highlighters', 'Kores Whitener Pen',
        'Solo Document Bag', 'Luxor Whiteboard Marker', 'Neelgagan Stamp Pad'
    ],
    'Pet Supplies': [
        'Pedigree Adult Dog Food 10kg', 'Royal Canin Maxi Puppy 4kg', 'Whiskas Cat Food Tuna 7kg',
        'Drools Focus Puppy Super Premium 4kg', 'Basil Rubber Chew Toy', 'PetSafe Gentle Leader Collar',
        'IRIS Pet Food Container', 'Fluval Aquarium Filter', 'Tetra Fish Food 1L',
        'Cat Mate Pet Fountain', 'Pet Life Lounger Bed', 'Dog Chow Adult Complete 8kg',
        'Farmina N&D Dog Food 12kg', 'PetSafe Drinkwell Water Fountain', 'Kong Classic Dog Toy',
        'Me-O Cat Food Seafood 7kg', 'Savic Dog Carrier', 'Pawzone Cat Scratch Post'
    ],
    'Paint and ancillaries': [
        'Asian Paints Royale Luxury Emulsion', 'Nerolac Impressions HD Paint', 'Berger Silk Luxury Emulsion',
        'Dulux Velvet Touch Diamond Glo', 'Asian Paints Ultima Protek Exterior', 'Nerolac Excel Anti-Peel Exterior',
        'Asian Paints Premium Emulsion', 'Berger WeatherCoat All Guard', 'Nerolac Beauty Gold Washable',
        'Asian Paints Tractor Emulsion', 'Asian Paints Apcolite Premium Enamel', 'Berger Easy Clean',
        'Asian Paints Ultra Adhesive', 'Asian Paints Dynamo Adhesive', 'Fevicol Marine Waterproof', 
        'Dr. Fixit Raincoat Waterproofing', 'Pidilite M-Seal', 'Fevicol SH Wood Adhesive 1kg'
    ]
}

# Generate base data
data = []
time_idx = 1

distributor_ids = [f"DIST{i:03d}" for i in range(1, NUM_DISTRIBUTORS + 1)]

# Create a list of all possible products
all_products = []
for category, products in PRODUCT_NAMES.items():
    for product in products:
        all_products.append((product, category))

# Randomly select NUM_SKUS products
selected_products = random.sample(all_products, min(NUM_SKUS, len(all_products)))
products_with_categories = [(product, category) for product, category in selected_products]

# Assign industry to each distributor
distributor_industry = {dist_id: random.choice(INDUSTRIES) for dist_id in distributor_ids}

# Create seasonal patterns for festivals
def is_diwali(year, quarter):
    # Diwali typically falls in Q4 (Oct-Nov)
    return 1 if quarter == 4 else 0

def is_ganesh_chaturthi(year, quarter):
    # Ganesh Chaturthi typically falls in Q3 (August-September)
    return 1 if quarter == 3 else 0

def is_gudi_padwa(year, quarter):
    # Gudi Padwa typically falls in Q1 or Q2 (March-April)
    return 1 if quarter in [1, 2] else 0

def is_eid(year, quarter):
    # Eid can occur in different quarters based on the Islamic calendar
    # This is a simplified version
    if year == 2020 and quarter in [2, 4]:
        return 1
    elif year == 2021 and quarter in [2, 3]:
        return 1
    elif year == 2022 and quarter in [1, 3]:
        return 1
    elif year == 2023 and quarter in [1, 3]:
        return 1
    elif year == 2024 and quarter in [1, 2]:
        return 1
    elif year == 2025 and quarter in [1, 2]:
        return 1
    return 0

def is_akshaya_tritiya(year, quarter):
    # Akshaya Tritiya typically falls in Q2 (April-May)
    return 1 if quarter == 2 else 0

def is_dussehra_navratri(year, quarter):
    # Dussehra/Navratri typically falls in Q3 or Q4 (September-October)
    return 1 if quarter in [3, 4] else 0

def is_onam(year, quarter):
    # Onam typically falls in Q3 (August-September)
    return 1 if quarter == 3 else 0

def is_christmas(year, quarter):
    # Christmas falls in Q4 (December)
    return 1 if quarter == 4 else 0

# Create baseline sales for each product
product_base_sales = {product: np.random.lognormal(mean=8, sigma=1) for product, _ in products_with_categories}

# Create seasonal multipliers for each quarter
quarter_multipliers = {
    1: 0.85,  # Q1 typically lower
    2: 1.0,   # Q2 neutral
    3: 1.1,   # Q3 slightly higher
    4: 1.3    # Q4 highest (holiday season)
}

# "Paint and ancillaries" category gets an extra boost in Q2 (spring/summer painting season)
paint_quarter_multipliers = {
    1: 0.75,  # Q1 lowest (winter)
    2: 1.5,   # Q2 highest (spring renovation season)
    3: 1.2,   # Q3 still good (summer projects)
    4: 0.9    # Q4 dropping (winter approaching)
}

# Create the data
all_time_series = []

for distributor_id in distributor_ids:
    industry = distributor_industry[distributor_id]
    
    # Each distributor carries a subset of products
    distributor_products = random.sample(products_with_categories, k=min(random.randint(20, 50), len(products_with_categories)))
    
    for product, category in distributor_products:
        # Base sales for this product-distributor combination with some randomness
        # Base sales for this product-distributor combination with some randomness
        distributor_product_factor = np.random.uniform(0.5, 1.5)
        base_sales = product_base_sales[product] * distributor_product_factor
        
        # Generate quarterly data
        for year in range(START_YEAR, END_YEAR + 1):
            for quarter in QUARTERS:
                # Skip some combinations to make the dataset more realistic
                if random.random() < 0.05:  # 5% chance to skip
                    continue
                
                # Calculate sales with seasonal patterns
                if category == 'Paint and ancillaries':
                    quarter_factor = paint_quarter_multipliers[quarter]
                else:
                    quarter_factor = quarter_multipliers[quarter]
                    
                festival_boost = 1.0
                
                # Add festival boosts
                if is_diwali(year, quarter):
                    festival_boost += 0.3 if category in ['Electronics', 'Apparel', 'Home Goods', 'Paint and ancillaries'] else 0.1

                if is_ganesh_chaturthi(year, quarter):
                    festival_boost += 0.25 if category in ['Food', 'Beverages'] else 0.05

                if is_gudi_padwa(year, quarter):
                    festival_boost += 0.2 if category in ['Apparel', 'Home Goods', 'Paint and ancillaries'] else 0.05

                if is_eid(year, quarter):
                    festival_boost += 0.25 if category in ['Food', 'Apparel'] else 0.05

                if is_akshaya_tritiya(year, quarter):
                    festival_boost += 0.2 if category in ['Jewelry', 'Home Goods'] else 0.05

                if is_dussehra_navratri(year, quarter):
                    festival_boost += 0.25 if category in ['Apparel', 'Home Goods', 'Electronics'] else 0.1

                if is_onam(year, quarter):
                    festival_boost += 0.2 if category in ['Food', 'Beverages'] else 0.05

                if is_christmas(year, quarter):
                    festival_boost += 0.3 if category in ['Electronics', 'Toys', 'Apparel'] else 0.1
                
                # Add yearly growth trend
                yearly_growth = 1 + 0.05 * (year - START_YEAR)
                
                # Add some noise
                noise = np.random.normal(1, 0.1)
                
                # Calculate final sales
                sales = base_sales * quarter_factor * festival_boost * yearly_growth * noise
                sales = max(0, sales)  # Ensure non-negative sales
                
                # Round to 2 decimal places
                sales = round(sales, 2)
                
                # Store the time series point with random movement category (placeholder for user input)
                time_point = {
                    'distributor_id': distributor_id,
                    'industry': industry,
                    'sku': product,  # Use actual product name instead of SKU ID
                    'category': category,
                    'quarter': quarter,
                    'year': year,
                    'sales': sales,
                    'is_diwali': is_diwali(year, quarter),
                    'is_ganesh_chaturthi': is_ganesh_chaturthi(year, quarter),
                    'is_gudi_padwa': is_gudi_padwa(year, quarter),
                    'is_eid': is_eid(year, quarter),
                    'is_akshay_tritiya': is_akshaya_tritiya(year, quarter),
                    'is_dussehra_navratri': is_dussehra_navratri(year, quarter),
                    'is_onam': is_onam(year, quarter),
                    'is_christmas': is_christmas(year, quarter),
                    'time_idx': time_idx,
                    'movement_category': random.choice(MOVEMENT_CATEGORIES)  # Random initial value to be replaced by user
                }
                
                all_time_series.append(time_point)
                time_idx += 1

# Convert to DataFrame
df = pd.DataFrame(all_time_series)

# Calculate derived fields
# Group by distributor, SKU, year, quarter to calculate total quarter sales
quarter_sales = df.groupby(['distributor_id', 'sku', 'year', 'quarter'])['sales'].sum().reset_index()
quarter_sales.rename(columns={'sales': 'total_quarter_sales'}, inplace=True)

# Merge back to original dataframe
df = pd.merge(df, quarter_sales, on=['distributor_id', 'sku', 'year', 'quarter'])

# Calculate average quarterly sales per SKU
avg_quarterly_sales = df.groupby(['sku'])['sales'].mean().reset_index()
avg_quarterly_sales.rename(columns={'sales': 'avg_quarterly_sales'}, inplace=True)
df = pd.merge(df, avg_quarterly_sales, on='sku')

# Calculate previous quarter sales (shift by 1 quarter)
df = df.sort_values(by=['distributor_id', 'sku', 'year', 'quarter'])
df['prev_quarter_sales'] = df.groupby(['distributor_id', 'sku'])['sales'].shift(1)

# Fill NaN values for first quarters
df['prev_quarter_sales'].fillna(df['sales'] * 0.9, inplace=True)  # Assume slight growth

# Ensure we have about 10000 rows
if len(df) > NUM_ROWS:
    df = df.sample(NUM_ROWS)
elif len(df) < NUM_ROWS:
    # Duplicate some rows with slight modifications to reach target size
    rows_needed = NUM_ROWS - len(df)
    extra_rows = df.sample(rows_needed, replace=True)
    extra_rows['sales'] = extra_rows['sales'] * np.random.uniform(0.95, 1.05, size=len(extra_rows))
    extra_rows['time_idx'] = range(df['time_idx'].max() + 1, df['time_idx'].max() + 1 + len(extra_rows))
    df = pd.concat([df, extra_rows])

# Final cleanup and formatting
df['sales'] = df['sales'].round(2)
df['avg_quarterly_sales'] = df['avg_quarterly_sales'].round(2)
df['total_quarter_sales'] = df['total_quarter_sales'].round(2)
df['prev_quarter_sales'] = df['prev_quarter_sales'].round(2)

# Reorder columns to match requested structure
df = df[['distributor_id', 'industry', 'sku', 'category', 'sales', 'avg_quarterly_sales', 
         'movement_category', 'quarter', 'year', 'total_quarter_sales', 'prev_quarter_sales',
         'is_diwali', 'is_ganesh_chaturthi', 'is_gudi_padwa', 'is_eid', 'is_akshay_tritiya', 
         'is_dussehra_navratri', 'is_onam', 'is_christmas', 'time_idx']]

# Save to CSV
df.to_csv('sales_data.csv', index=False)

print(f"Dataset successfully saved to 'sales_dataset.csv'")
print(f"Dataset shape: {df.shape}")
print(f"Number of distributors: {df['distributor_id'].nunique()}")
print(f"Number of unique products: {df['sku'].nunique()}")
print(f"Years covered: {df['year'].min()} to {df['year'].max()}")
print(f"Number of industries: {df['industry'].nunique()}")
print(f"Number of categories: {df['category'].nunique()}")
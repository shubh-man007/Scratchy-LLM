import sqlite3

conn = sqlite3.connect("hotels.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS hotels (
    id TEXT PRIMARY KEY,
    name TEXT,
    location TEXT,
    price_tier TEXT,
    checkin_date TEXT,
    checkout_date TEXT,
    booked INTEGER    
)
""")

cursor.execute("INSERT OR REPLACE INTO hotels VALUES (1, 'Hilton Basel', 'Basel', 'Luxury', '2024-04-22', '2024-04-20', 0)")
cursor.execute("INSERT OR REPLACE INTO hotels VALUES (2, 'Marriott Zurich', 'Zurich', 'Upscale', '2024-04-14', '2024-04-21', 0)")
cursor.execute("INSERT OR REPLACE INTO hotels VALUES (3, 'Hyatt Regency Basel', 'Basel', 'Upper Upscale', '2024-04-02', '2024-04-20', 0)")
cursor.execute("INSERT OR REPLACE INTO hotels VALUES (4, 'Radisson Lucerne', 'Lucerne', 'Midscale', '2024-04-24', '2024-04-05', 0)")
cursor.execute("INSERT OR REPLACE INTO hotels VALUES (5, 'Best Western Bern', 'Bern', 'Upper Midscale', '2024-04-23', '2024-04-01', 0)")
cursor.execute("INSERT OR REPLACE INTO hotels VALUES (6, 'InterContinental Geneva', 'Geneva', 'Luxury', '2024-04-23', '2024-04-28', 0)")
cursor.execute("INSERT OR REPLACE INTO hotels VALUES (7, 'Sheraton Zurich', 'Zurich', 'Upper Upscale', '2024-04-27', '2024-04-02', 0)")
cursor.execute("INSERT OR REPLACE INTO hotels VALUES (8, 'Holiday Inn Basel', 'Basel', 'Upper Midscale', '2024-04-24', '2024-04-09', 0)")
cursor.execute("INSERT OR REPLACE INTO hotels VALUES (9, 'Courtyard Zurich', 'Zurich', 'Upscale', '2024-04-03', '2024-04-13', 0)")
cursor.execute("INSERT OR REPLACE INTO hotels VALUES (10, 'Comfort Inn Bern', 'Bern', 'Midscale', '2024-04-04', '2024-04-16', 0)")

conn.commit()
conn.close()


# Run commands:
# ./toolbox --tools-file "tools.yaml"  

# Output:
# (1, 'Hilton Basel', 'Basel', 'Luxury', '2024-04-22', '2024-04-20', 0)
# (2, 'Marriott Zurich', 'Zurich', 'Upscale', '2024-04-14', '2024-04-21', 0)
# (3, 'Hyatt Regency Basel', 'Basel', 'Upper Upscale', '2024-04-02', '2024-04-20', 0)
# (4, 'Radisson Lucerne', 'Lucerne', 'Midscale', '2024-04-24', '2024-04-05', 0)
# (5, 'Best Western Bern', 'Bern', 'Upper Midscale', '2024-04-23', '2024-04-01', 0)
# (6, 'InterContinental Geneva', 'Geneva', 'Luxury', '2024-04-23', '2024-04-28', 0)
# (7, 'Sheraton Zurich', 'Zurich', 'Upper Upscale', '2024-04-27', '2024-04-02', 0)
# (8, 'Holiday Inn Basel', 'Basel', 'Upper Midscale', '2024-04-24', '2024-04-09', 0)
# (9, 'Courtyard Zurich', 'Zurich', 'Upscale', '2024-04-03', '2024-04-13', 0)
# (10, 'Comfort Inn Bern', 'Bern', 'Midscale', '2024-04-04', '2024-04-16', 0)

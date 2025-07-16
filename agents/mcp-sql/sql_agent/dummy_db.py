import sqlite3

conn = sqlite3.connect("database.db")
cursor = conn.cursor()

cursor.execute("PRAGMA foreign_keys = ON;")

cursor.executescript("""
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS products;
""")

cursor.executescript("""
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT,
    city TEXT
);

CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    price REAL NOT NULL,
    category TEXT
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    product_id INTEGER,
    quantity INTEGER,
    order_date TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);
""")

cursor.executemany(
    "INSERT INTO customers (name, email, city) VALUES (?, ?, ?);",
    [
        ("Alice Smith", "alice@example.com", "New York"),
        ("Bob Johnson", "bob@example.com", "Los Angeles"),
        ("Charlie Lee", "charlie@example.com", "Chicago"),
    ]
)

cursor.executemany(
    "INSERT INTO products (name, price, category) VALUES (?, ?, ?);",
    [
        ("Laptop", 1200.00, "Electronics"),
        ("Desk Chair", 150.00, "Furniture"),
        ("Wireless Mouse", 30.00, "Electronics"),
    ]
)


cursor.executemany(
    "INSERT INTO orders (customer_id, product_id, quantity, order_date) VALUES (?, ?, ?, ?);",
    [
        (1, 1, 1, "2024-05-01"),  
        (2, 2, 2, "2024-05-03"),  
        (3, 3, 3, "2024-05-05"),  
        (1, 3, 1, "2024-05-07"),  
    ]
)


conn.commit()
conn.close()

print("Dummy database created as 'database.db'")

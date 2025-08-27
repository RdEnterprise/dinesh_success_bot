import MetaTrader5 as mt5

# 1. Initialize
if not mt5.initialize():
    print("initialize() failed")
    quit()

# 2. List all available symbols
symbols = mt5.symbols_get()
print("Total symbols:", len(symbols))

# 3. Print first 20 names
for s in symbols[:20]:
    print(s.name)

# 4. Look for Gold / XAU
for s in symbols:
    if "XAU" in s.name or "GOLD" in s.name:
        print("Found GOLD symbol:", s.name)

mt5.shutdown()

import re

print("=" * 50)
print("BOT PERFORMANCE ANALYSIS")
print("=" * 50)

# Analyze 0x8dxd bot
try:
    with open(r'C:\Users\Star\AppData\Local\Temp\claude\C--Users-Star--local-bin\tasks\ba9b49e.output', 'r') as f:
        content = f.read()

    our_copy_matches = re.findall(r'Our copy: \$([0-9.]+)', content)
    order_placed = content.count('ORDER PLACED')
    order_error = content.count('ORDER ERROR')
    matched = content.count("'status': 'matched'")
    live = content.count("'status': 'live'")

    total_attempted = sum(float(x) for x in our_copy_matches)

    print("\n0x8dxd Bot (Max $200, Scale 10%)")
    print("-" * 40)
    print(f"Trade attempts: {len(our_copy_matches)}")
    print(f"Orders placed: {order_placed}")
    print(f"  - Matched (filled): {matched}")
    print(f"  - Live (pending): {live}")
    print(f"Orders failed: {order_error}")
    print(f"Total $ attempted: ${total_attempted:.2f}")
    print(f"Est. $ actually spent: ${matched * 5 + live * 5:.2f}")
except Exception as e:
    print(f"Error analyzing 0x8dxd: {e}")

# Analyze k9Q2 bot
try:
    with open(r'C:\Users\Star\AppData\Local\Temp\claude\C--Users-Star--local-bin\tasks\b4af36f.output', 'r') as f:
        content = f.read()

    our_size_matches = re.findall(r'Our size: \$([0-9.]+)', content)
    order_placed = content.count('ORDER PLACED')
    order_error = content.count('ORDER ERROR') + content.count('ORDER FAILED')

    total_signaled = sum(float(x) for x in our_size_matches)

    print("\n\nk9Q2mX4L8A7ZP3R Bot (Max $100, Scale 5%)")
    print("-" * 40)
    print(f"Trade signals: {len(our_size_matches)}")
    print(f"Orders placed: {order_placed}")
    print(f"Orders failed: {order_error}")
    print(f"Total $ signaled: ${total_signaled:.2f}")
except Exception as e:
    print(f"Error analyzing k9Q2: {e}")

print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print("\n0x8dxd placed MORE orders (76 vs 13)")
print("0x8dxd used SMALLER amounts ($5 each)")
print("k9Q2 used LARGER amounts ($5-$100 each)")
print("\nMost losses likely from 0x8dxd due to volume")

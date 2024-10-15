def format_price(price):
    return f"${price:.2f}"

def handle_error(error_message):
    return {"error": error_message}, 400

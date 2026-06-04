import random
import string
import json
from datetime import datetime, timedelta

KEY_POOL = [
    "id", "user_id", "uuid", "session_id", "trace_id", "request_id",
    "name", "username", "first_name", "last_name", "full_name",
    "email", "phone", "password_hash", "auth_token",
    "profile", "account", "settings", "preferences", "metadata",
    "status", "role", "permissions", "group", "tier", "plan",
    "address", "street", "city", "state", "country", "postcode",
    "latitude", "longitude", "geo", "location",
    "timestamp", "created_at", "updated_at", "deleted_at",
    "start_time", "end_time", "duration", "event_time",
    "log_level", "message", "event", "event_type",
    "request", "response", "payload", "headers", "body",
    "endpoint", "route", "method", "query", "params",
    "status_code", "error", "error_code", "error_message",
    "data", "value", "values", "count", "total", "sum",
    "average", "min", "max", "score", "metric", "metrics",
    "features", "embedding", "vector",
    "items", "list", "array", "results", "records",
    "entries", "children", "nodes", "edges",
    "config", "configuration", "version", "build", "environment",
    "service", "service_name", "host", "ip", "port",
    "region", "cluster", "node", "instance",
    "product", "product_id", "price", "currency", "quantity",
    "order", "order_id", "cart", "invoice", "payment",
    "transaction", "amount", "discount", "tax",
    "enabled", "disabled", "active", "inactive",
    "success", "failure", "pending", "retry",
    "flag", "is_valid", "is_deleted",
    "model", "prediction", "label", "confidence",
    "input", "output", "target", "loss",
    "embedding_dim", "attention", "layer",
    "debug", "trace", "stacktrace", "warning",
    "notes", "comment", "description", "tags",
    "category", "type", "kind", "source", "origin"
]

VALUE_GENERATORS = [
    lambda: random.randint(0, 10000),
    lambda: round(random.random() * 1000, random.randint(1, 5)),
    lambda: "".join(random.choices(string.ascii_letters, k=random.randint(3, 12))),
    lambda: random.choice([True, False]),
    lambda: None
]

def generate_value(depth, max_depth):
    """Generate nested JSON values with controlled complexity."""

    # Stop recursion
    if depth >= max_depth:
        return random.choice(VALUE_GENERATORS)()

    choice = random.random()

    # Primitive value
    if choice < 0.4:
        return random.choice(VALUE_GENERATORS)()

    # Nested object
    elif choice < 0.75:
        return generate_object(depth + 1, max_depth)

    # Array
    else:
        return generate_array(depth + 1, max_depth)


def generate_object(depth=0, max_depth=4, max_keys=6):
    """Generate a random JSON object."""

    obj = {}
    num_keys = random.randint(1, max_keys)
    randCount = 0
    
    for item in range(max_keys):
        randCount += 1
        if random.randint(1,2) == 1:
            break

    num_keys = randCount

    for _ in range(num_keys):
        key = random.choice(KEY_POOL)

        # Ensure some uniqueness pressure
        if random.random() < 0.3:
            key += "_" + "".join(random.choices(string.ascii_lowercase, k=3))

        obj[key] = generate_value(depth, max_depth)

    # Occasionally inject missing fields
    if random.random() < 0.2 and len(obj) > 1:
        obj.pop(random.choice(list(obj.keys())))

    return obj


def generate_array(depth=0, max_depth=4, max_len=8):
    """Generate a JSON array with mixed structures."""

    length = random.randint(1, max_len)
    arr = []

    for _ in range(length):
        arr.append(generate_value(depth, max_depth))

    return arr


def generate_complex_json(num_samples=1, max_depth=5, max_keys=6):
    """Generate a list of complex JSON objects."""

    samples = []

    for _ in range(num_samples):
        
        obj = generate_object(0, max_depth, 5)
        
        samples.append(obj)

    return samples
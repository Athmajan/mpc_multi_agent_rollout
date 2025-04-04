import time

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.time()  # Record the end time
        print(f"Execution time for {func.__name__}: {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# Example usage
@measure_time
def example_function(n):
    total = 0
    for i in range(n):
        total += i ** 2
    return total

# Call the decorated function
result = example_function(1000000)
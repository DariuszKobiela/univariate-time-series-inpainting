import random
import string
import hashlib


def generate_random_string(length=16, use_uppercase=True, use_lowercase=True, 
                          use_digits=True, use_special_chars=False):
    """
    Generuje losowy ciąg znaków o określonej długości.
    
    Args:
        length (int): Długość generowanego ciągu znaków (domyślnie 16)
        use_uppercase (bool): Czy używać wielkich liter (domyślnie True)
        use_lowercase (bool): Czy używać małych liter (domyślnie True)
        use_digits (bool): Czy używać cyfr (domyślnie True)
        use_special_chars (bool): Czy używać znaków specjalnych (domyślnie False)
    
    Returns:
        str: Losowy ciąg znaków
    """
    characters = ""
    
    if use_lowercase:
        characters += string.ascii_lowercase
    if use_uppercase:
        characters += string.ascii_uppercase
    if use_digits:
        characters += string.digits
    if use_special_chars:
        characters += "!@#$%^&*()_+-=[]{}|;:,.<>?"
    
    if not characters:
        raise ValueError("Przynajmniej jeden typ znaków musi być włączony")
    
    return ''.join(random.choice(characters) for _ in range(length))


def string_to_hash(input_string, hash_algorithm='sha256'):
    """
    Transformuje ciąg znaków na hash przy użyciu określonego algorytmu.
    
    Args:
        input_string (str): Ciąg znaków do zahashowania
        hash_algorithm (str): Algorytm hashowania ('md5', 'sha1', 'sha256', 'sha512')
    
    Returns:
        str: Hash w postaci heksadecymalnej
    """
    # Konwersja na bytes
    encoded_string = input_string.encode('utf-8')
    
    # Wybór algorytmu hashowania
    if hash_algorithm.lower() == 'md5':
        hash_object = hashlib.md5(encoded_string)
    elif hash_algorithm.lower() == 'sha1':
        hash_object = hashlib.sha1(encoded_string)
    elif hash_algorithm.lower() == 'sha256':
        hash_object = hashlib.sha256(encoded_string)
    elif hash_algorithm.lower() == 'sha512':
        hash_object = hashlib.sha512(encoded_string)
    else:
        raise ValueError(f"Nieobsługiwany algorytm hashowania: {hash_algorithm}")
    
    return hash_object.hexdigest()


def generate_random_string_with_hash(length=16, hash_algorithm='sha256', 
                                   use_uppercase=True, use_lowercase=True, 
                                   use_digits=True, use_special_chars=False):
    """
    Generuje losowy ciąg znaków i zwraca zarówno oryginalny ciąg jak i jego hash.
    
    Args:
        length (int): Długość generowanego ciągu znaków
        hash_algorithm (str): Algorytm hashowania
        use_uppercase (bool): Czy używać wielkich liter
        use_lowercase (bool): Czy używać małych liter
        use_digits (bool): Czy używać cyfr
        use_special_chars (bool): Czy używać znaków specjalnych
    
    Returns:
        tuple: (oryginalny_ciąg, hash_ciągu)
    """
    random_string = generate_random_string(
        length=length,
        use_uppercase=use_uppercase,
        use_lowercase=use_lowercase,
        use_digits=use_digits,
        use_special_chars=use_special_chars
    )
    
    hash_value = string_to_hash(random_string, hash_algorithm)
    
    return random_string, hash_value


# Przykład użycia
if __name__ == "__main__":
    print("=== Generator losowych ciągów znaków z hashowaniem ===\n")
    
    # Przykład 1: Podstawowe użycie
    print("1. Podstawowe użycie:")
    random_str, hash_val = generate_random_string_with_hash()
    print(f"   Losowy ciąg: {random_str}")
    print(f"   SHA256 hash: {hash_val}")
    print()
    
    # Przykład 2: Długi ciąg z różnymi algorytmami
    print("2. Różne algorytmy hashowania:")
    test_string = generate_random_string(20)
    print(f"   Ciąg testowy: {test_string}")
    
    for algorithm in ['md5', 'sha1', 'sha256', 'sha512']:
        hash_result = string_to_hash(test_string, algorithm)
        print(f"   {algorithm.upper()}: {hash_result}")
    print()
    
    # Przykład 3: Ze znakami specjalnymi
    print("3. Z użyciem znaków specjalnych:")
    special_str, special_hash = generate_random_string_with_hash(
        length=24, 
        use_special_chars=True,
        hash_algorithm='sha512'
    )
    print(f"   Losowy ciąg: {special_str}")
    print(f"   SHA512 hash: {special_hash}")
    print()
    
    # Przykład 4: Tylko cyfry
    print("4. Tylko cyfry:")
    digit_str, digit_hash = generate_random_string_with_hash(
        length=12,
        use_uppercase=False,
        use_lowercase=False,
        use_digits=True,
        hash_algorithm='md5'
    )
    print(f"   Losowy ciąg: {digit_str}")
    print(f"   MD5 hash: {digit_hash}") 
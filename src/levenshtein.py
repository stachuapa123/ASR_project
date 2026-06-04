def lev_weighted(s1, s2, 
                          cost_delete=0.5,      # taniej usuwać
                          cost_insert=1.0,
                          cost_substitute=1.0):
    n, m = len(s1), len(s2)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        dp[i][0] = i * cost_delete
    for j in range(m + 1):
        dp[0][j] = j * cost_insert
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + cost_delete,
                    dp[i][j-1] + cost_insert,
                    dp[i-1][j-1] + cost_substitute,
                )
    
    return dp[n][m]

def damerau_lev(s1, s2):
    """Klasyczny DL — operacje insert/delete/substitute/transpose, każda kosztuje 1."""
    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            
            dp[i][j] = min(
                dp[i-1][j] + 1,           # delete
                dp[i][j-1] + 1,           # insert
                dp[i-1][j-1] + cost,      # substitute (lub match)
            )
            
            # NEW — transpozycja
            if i >= 2 and j >= 2 \
               and s1[i-1] == s2[j-2] \
               and s1[i-2] == s2[j-1]:
                dp[i][j] = min(dp[i][j], dp[i-2][j-2] + 1)
    
    return dp[n][m]

def true_damerau_levenshtein(s1, s2):
    """Prawdziwy Damerau-Levenshtein (nieograniczone transpozycje)."""
    INF = float('inf')
    da = {}                  # last occurrence of each character
    
    n, m = len(s1), len(s2)
    # tablica (n+2) x (m+2), z sentinel wartościami
    H = [[0] * (m + 2) for _ in range(n + 2)]
    
    max_dist = n + m
    H[0][0] = max_dist
    for i in range(0, n + 1):
        H[i + 1][0] = max_dist
        H[i + 1][1] = i
    for j in range(0, m + 1):
        H[0][j + 1] = max_dist
        H[1][j + 1] = j
    
    for i in range(1, n + 1):
        db = 0
        for j in range(1, m + 1):
            k = da.get(s2[j - 1], 0)
            l = db
            cost = 1
            if s1[i - 1] == s2[j - 1]:
                cost = 0
                db = j
            
            H[i + 1][j + 1] = min(
                H[i][j] + cost,                                  # substitute
                H[i + 1][j] + 1,                                 # insert
                H[i][j + 1] + 1,                                 # delete
                H[k][l] + (i - k - 1) + 1 + (j - l - 1),         # transpose
            )
        da[s1[i - 1]] = i
    
    return H[n + 1][m + 1]

PHONEME_GROUPS = {
    'vowels':        {'a', 'e', 'i', 'i2', 'o', 'u'},
    'plosives':      {'p', 'b', 't', 'd', 'k', 'g', 'c'},
    'fricatives':    {'f', 'v', 's', 'z', 'S', 'Z', 'sj', 'zj', 'h'},
    'affricates':    {'dz', 'tS', 'dZ', 'tsj', 'dzj'},
    'nasals':        {'m', 'n', 'n~'},
    'liquids':       {'l', 'r'},
    'glides':        {'j', 'w'},
}

VERY_SIMILAR = [
    {'s', 'z'}, {'p', 'b'}, {'t', 'd'}, {'k', 'g'}, {'f', 'v'},
    {'S', 'Z'}, {'tS', 'dZ'}, {'tsj', 'dzj'}, {'sj', 'zj'},
    {'sj', 'S'}, {'tsj', 'tS'},
    {'a', 'o'}, {'e', 'i2'}, {'i', 'i2'},
]


PHONEME_GROUPS2 = {
    'vowels':        {'a', 'e', 'i', 'o', 'u', 'y', 'ą', 'ę', 'ó'},
    'plosives':      {'p', 'b', 't', 'd', 'k', 'g', 'c'},
    'fricatives':    {'f', 'v', 's', 'z', 'S', 'ż', 'ś', 'ź', 'h'},
    #'affricates':    {'dz', 'tS', 'dZ', 'tsj', 'dzj'},
    'nasals':        {'m', 'n', 'ń'},
    'liquids':       {'l', 'r'},
    'glides':        {'j', 'ł'},
}

VERY_SIMILAR2 = [
    {'s', 'z'}, {'p', 'b'}, {'t', 'd'}, {'k', 'g'}, {'f', 'w'},
    {'sz', 'rz'}, {'tS', 'dZ'}, {'tsj', 'dzj'}, {'ś', 'ź'},
    {'ś', 'sz'}, {'tsj', 'tS'},
    {'a', 'o'}, {'e', 'y'}, {'i', 'y'}, {'i', 'j'},
    {'a', 'e'}, {'a', 'ą'}, {'e', 'ę'}
]

ORTHOGRAPHY = [
    {'u', 'ó'}, {'ch', 'h'}, ('rz', 'ż')
]

def phoneme_substitution_cost(a, b):
    if a == b:
        return 0.0
    if {a, b} in ORTHOGRAPHY:
        return 0.0
    if {a, b} in VERY_SIMILAR:
        return 0.3
    for group in PHONEME_GROUPS2.values():
        if a in group and b in group:
            return 0.5
    return 1.0

def substitution_cost(a, b):
    """Koszt zamiany 'a' na 'b'. Zwraca 0 dla identycznych."""
    if a == b:
        return 0.0
    
    # samogłoski mylą się ze sobą — mały koszt
    vowels = set('aeiouyąęóy')
    if a in vowels and b in vowels:
        return 0.3
    
    # spółgłoski o podobnym miejscu artykulacji
    similar_pairs = [
        {'p', 'b'}, {'t', 'd'}, {'k', 'g'},      # zwarte (voiced/voiceless)
        {'f', 'v'}, {'s', 'z'}, {'sz', 'ż'},     # frykatywy
        {'sz', 'ś'}, {'cz', 'ć'},                 # twarde/miękkie
        {'i', 'j'}, {'ji', 'i'}                            # ortograficzne warianty
    ]

    orthographic_pairs = [
        {'rz', 'ż'}, {'u', 'ó'}, {'h', 'ch'} 
    ]
    for pair in similar_pairs:
        if {a, b} == pair:
            return 0.5
        
    for pair in orthographic_pairs:
        if {a,b} == pair:
            return 0
    
    # różne kategorie — pełny koszt
    return 1.0

def damerau_levenshtein_weighted(s1, s2,
                                   cost_delete=0.3,
                                   cost_insert=0.8,
                                   cost_transpose=0.5,
                                   sub_cost_fn=phoneme_substitution_cost):
    """Damerau-Levenshtein z różnymi kosztami operacji + transpozycja."""
    n, m = len(s1), len(s2)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        dp[i][0] = i * cost_delete
    for j in range(m + 1):
        dp[0][j] = j * cost_insert
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = sub_cost_fn(s1[i-1], s2[j-1])
            
            dp[i][j] = min(
                dp[i-1][j] + cost_delete,
                dp[i][j-1] + cost_insert,
                dp[i-1][j-1] + sub,
            )
            
            # transpozycja sąsiednich znaków
            if i >= 2 and j >= 2 \
               and s1[i-1] == s2[j-2] \
               and s1[i-2] == s2[j-1]:
                dp[i][j] = min(dp[i][j], dp[i-2][j-2] + cost_transpose)
    
    return dp[n][m]





def levenshtein_phoneme_aware(s1, s2,
                                cost_delete=0.5,
                                cost_insert=1.0,
                                sub_cost_fn=substitution_cost):
    n, m = len(s1), len(s2)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        dp[i][0] = i * cost_delete
    for j in range(m + 1):
        dp[0][j] = j * cost_insert
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = sub_cost_fn(s1[i-1], s2[j-1])
            
            dp[i][j] = min(
                dp[i-1][j] + cost_delete,
                dp[i][j-1] + cost_insert,
                dp[i-1][j-1] + sub,
            )
    
    return dp[n][m]
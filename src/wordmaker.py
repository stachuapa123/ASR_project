PHONEME_TO_LETTERS = {
    # 1:1 mapping
    'a': 'a', 'e': 'e', 'o': 'o', 'u': 'u', 'i': 'i', 'i2': 'y',
    'p': 'p', 'b': 'b', 't': 't', 'd': 'd', 'k': 'k', 'g': 'g',
    'f': 'f', 'v': 'w', 's': 's', 'z': 'z', 'm': 'm', 'n': 'n',
    'l': 'l', 'r': 'r', 'j': 'j', 'w': 'ł', 'h': 'h',
    
    # specjalne polskie
    'S': 'sz', 'Z': 'ż', 'tS': 'cz', 'dZ': 'dż',
    'c': 'c', 'dz': 'dz',
    
    # miękkie — wymagają kontekstu (zobacz funkcję poniżej)
    'sj': 'ś', 'zj': 'ź', 'tsj': 'ć', 'dzj': 'dź', 'n~': 'ń',
    
    # nosówki
    'eo5': 'ę', 'oc5': 'ą',
    
    # cisza
    'sil': '', 'sp': '',
}


def phonemes_to_text(phonemes, after_silence = False):
    """Konwertuje listę fonemów na tekst z polskimi zasadami."""
    out = []
    for i, ph in enumerate(phonemes):
        next_ph = phonemes[i+1] if i+1 < len(phonemes) else None
          
        # miękkie spółgłoski przed samogłoską: "ś" → "si" + samogłoska
        # ale na końcu lub przed spółgłoską: zostają jako "ś"
        if ph in ('sj', 'zj', 'tsj', 'dzj', 'n~'):
            soft_map = {'sj': 'ś', 'zj': 'ź', 'tsj': 'ć', 'dzj': 'dź', 'n~': 'ń'}
            if next_ph == 'i':
                # "ś" + "i" → "si" (na piśmie)
                base = {'sj': 's', 'zj': 'z', 'tsj': 'c', 'dzj': 'dz', 'n~': 'n'}[ph]
                out.append(base)
            elif next_ph in ('a', 'e', 'o', 'u', 'eo5', 'oc5'):
                # "ś" + "a" → "sia" (na piśmie)
                base = {'sj': 's', 'zj': 'z', 'tsj': 'c', 'dzj': 'dz', 'n~': 'n'}[ph]
                out.append(base + 'i')
                # samogłoska zostanie dodana w następnej iteracji
            else:
                out.append(soft_map[ph])
        else:
            out.append(PHONEME_TO_LETTERS.get(ph, ph))
        if after_silence == False and next_ph == 'sil' and i > 0: #po sil juz nic nie liczy
            break
    return ''.join(out)

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]
"""
Konwertuje TextGridy z MFA (IPA) na twój custom zestaw fonemów.
Działa na całych folderach rekursywnie.
"""

import re
from pathlib import Path


# ─── MAPOWANIE IPA → twój zestaw ────────────────────────────────────────
IPA_TO_YOURS = {
    # ─── Samogłoski ───
    'a':   'a',
    'ɛ':   'e',           # [ɛ] to polskie "e"
    'e':   'e',
    'i':   'i',
    'ɨ':   'i2',          # [ɨ] to polskie "y"
    'ɔ':   'o',           # [ɔ] to polskie "o"
    'o':   'o',
    'u':   'u',
    'ɛ̃':   'eo5',         # nosowe e (samogłoska "ę")
    'ɔ̃':   'oc5',         # nosowe o (samogłoska "ą")
    
    # ─── Spółgłoski zwarte (twarde) ───
    'p':   'p',
    'b':   'b',
    't':   't',
    't̪':   't',           # zębowe t (z diakrytyką dentalną)
    'd':   'd',
    'd̪':   'd',
    'k':   'k',
    'ɡ':   'g',           # IPA g (różny znak niż 'g'!)
    'g':   'g',
    'c':   'c',           # 'c' w twoim zestawie - palatalne k
    
    # ─── Spółgłoski palatalizowane (miękkie) ───
    # Twój zestaw nie rozróżnia, więc mapujemy na zwykłe wersje
    'pʲ':  'p',
    'bʲ':  'b',
    'mʲ':  'm',
    'vʲ':  'v',
    'fʲ':  'f',
    
    # ─── Frykatywy ───
    'f':   'f',
    'v':   'v',
    's':   's',
    's̪':   's',
    'z':   'z',
    'z̪':   'z',
    'ʂ':   'S',           # sz
    'ʐ':   'Z',           # ż / rz
    'ɕ':   'sj',          # ś
    'ʑ':   'zj',          # ź
    'x':   'h',           # ch / h
    'h':   'h',
    
    # ─── Afrykaty ───
    't͡s':  'c',           # c (twarde, z tie-bar)
    'ts':  'c',           # bez tie-bara
    't̪s̪':  'c',           # zębowe ts (twoja wersja MFA)
    'd͡z':  'dz',          # dz
    'dz':  'dz',
    'd̪z̪':  'dz',
    't͡ʂ':  'tS',          # cz
    'tʂ':  'tS',
    'd͡ʐ':  'dZ',          # dż
    'dʐ':  'dZ',
    't͡ɕ':  'tsj',         # ć
    'tɕ':  'tsj',
    'd͡ʑ':  'dzj',         # dź
    'dʑ':  'dzj',
    
    # ─── Nosowe ───
    'm':   'm',
    'n':   'n',
    'n̪':   'n',
    'ɲ':   'n~',          # ń
    'ŋ':   'n',           # tylne n (allofon przed k/g)
    
    # ─── Płynne i półsamogłoski ───
    'l':   'l',
    'ʎ':   'l',           # palatalne l - w polskim rzadkie, mapuję na l
    'r':   'r',
    'j':   'j',
    'w':   'w',           # ł
    'rʲ':  'r',           # miękkie r (nie występuje w polskim, mapuję na twarde)
    
    # ─── Specjalne ───
    '':    'sil',            # cisza między fonemami (zachowujemy puste)
    'sil': 'sil',
    'sp':  'sp',
    'spn': 'sp',          # spoken noise
    '<unk>': 'sp',        # unknown word
}


def convert_textgrid_text(text: str, mapping: dict) -> tuple[str, set]:
    """Konwertuje string TextGrida — zamienia fonemy IPA na twój zestaw.
    Zwraca (skonwertowany_tekst, zbior_niemapowanych_fonemow)."""
    
    unmapped = set()
    
    def replace(match):
        original = match.group(1)
        if original in mapping:
            return f'text = "{mapping[original]}"'
        elif original == '':
            return f'text = ""'
        else:
            unmapped.add(original)
            return f'text = "{original}"'    # zostaw oryginał
    
    new_text = re.sub(r'text = "([^"]*)"', replace, text)
    return new_text, unmapped


def convert_file(input_path: Path, output_path: Path, mapping: dict) -> set:
    """Konwertuje jeden plik TextGrid. Zwraca zbiór niemapowanych fonemów."""
    
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    new_text, unmapped = convert_textgrid_text(text, mapping)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(new_text)
    
    return unmapped


def convert_folder(input_dir: str, output_dir: str, mapping: dict = None):
    """Konwertuje wszystkie .TextGrid w folderze (rekursywnie).
    
    Args:
        input_dir: folder z TextGridami w IPA
        output_dir: folder docelowy (struktura folderów zachowana)
        mapping: słownik IPA→fonemy (default: IPA_TO_YOURS)
    """
    
    if mapping is None:
        mapping = IPA_TO_YOURS
    
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    
    tg_files = list(input_dir.rglob('*.TextGrid'))
    print(f'znalazłem {len(tg_files)} plików TextGrid w {input_dir}')
    
    all_unmapped = set()
    converted = 0
    
    for tg_path in tg_files:
        # zachowaj strukturę folderów
        relative = tg_path.relative_to(input_dir)
        out_path = output_dir / relative
        
        try:
            unmapped = convert_file(tg_path, out_path, mapping)
            all_unmapped.update(unmapped)
            converted += 1
        except Exception as e:
            print(f'błąd przy {tg_path}: {e}')
    
    print(f'\nskonwertowano {converted}/{len(tg_files)} plików')
    print(f'zapisano do: {output_dir}')
    
    if all_unmapped:
        print(f'\n⚠️  NIEMAPOWANE FONEMY (zostały w oryginale):')
        for ph in sorted(all_unmapped):
            chars = ' '.join(f'U+{ord(c):04X}' for c in ph)
            print(f'  {ph!r:20s}  {chars}')
        print('\nDodaj te fonemy do IPA_TO_YOURS i uruchom ponownie.')
    else:
        print('\n✓ wszystkie fonemy zostały zmapowane')


# ─── UŻYCIE ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    convert_folder(
        input_dir='./AutorskieDane/BrudneGridy',       # folder z MFA
        output_dir='./AutorskieDane/AutorskiDataset',    # folder docelowy
    )
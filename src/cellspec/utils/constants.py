"""Constants used throughout cellspec."""

MUTATION_TYPES = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']

MUTATION_COLORS = {
    'C>A': '#00BCD4',  # Cyan
    'C>G': '#111111',  # Black
    'C>T': '#E62725',  # Red
    'T>A': '#D3D3D3',  # Light gray
    'T>C': '#99CC00',  # Green
    'T>G': '#FFADBA'   # Pink
}

TRINUC_CONTEXTS = ["A", "C", "G", "T"]


def get_canonical_96_order():
    """
    Generate the canonical 96 mutation context order used in COSMIC signatures.

    Returns 96 mutation types in the format: XYZ>XWZ where Y is the reference
    pyrimidine (C or T) and W is the alternate base, with X and Z being any base.

    Returns
    -------
    list of str
        List of 96 mutation contexts in canonical order

    Examples
    --------
    >>> contexts = get_canonical_96_order()
    >>> len(contexts)
    96
    >>> contexts[0]
    'ACA>AAA'
    """
    bases = ["A", "C", "G", "T"]
    substitutions = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
    contexts = []
    for sub in substitutions:
        ref_from, ref_to = sub.split(">")
        for b1 in bases:
            for b3 in bases:
                contexts.append(f"{b1}{ref_from}{b3}>{b1}{ref_to}{b3}")
    return contexts

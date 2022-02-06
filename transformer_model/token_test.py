from get_reaction_results import synthesis
from onmt.opts_translate import OPT_TRANSLATE


def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


test_smiles = 'c1ccc(Nc2nccc(-c3cnccn3)n2)cc1.OCc1[n+]cc(Nc2nccc(-c3ccncc3)n2)cc1', 'c1ccc(Nc2nccc(-c3cnccn3)n2)cc1.OCc1[n+]cc(Nc2nccc(-c3ccncc3)n2)cc1'

token_list = []
for smi in test_smiles:
    token_smi = smi_tokenizer(smi)
    token_list.append(token_smi)
opt = OPT_TRANSLATE()
synthesis(opt, token_list)

# print(token_list)

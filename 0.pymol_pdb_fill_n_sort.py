python
stored.i = 0
stored.last = None
def renumber_sequential(selection="all"):
    cmd.alter(selection, """
if (chain, resi, inscode) != stored.last:
    stored.i += 1
    stored.last = (chain, resi, inscode)
resi = str(stored.i)
""")
    cmd.sort(selection)
python end

renumber_sequential("chain A")
save mcl1_clean_renum.pdb, chain A
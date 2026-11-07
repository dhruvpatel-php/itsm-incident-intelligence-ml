from pathlib import Path
import nbformat
from IPython.core.interactiveshell import InteractiveShell

def test_notebook_execution():
    root = Path(__file__).resolve().parent.parent
    nb_path = root / "PRCL-0012_ITSM_Machine_Learning.ipynb"
    print(f"Reading notebook: {nb_path}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    shell = InteractiveShell.instance()
    
    code_cells = [cell for cell in nb.cells if cell.cell_type == 'code']
    print(f"Executing {len(code_cells)} code cells in-process...")
    
    for idx, cell in enumerate(code_cells):
        src = cell.source
        res = shell.run_cell(src)
        if not res.success:
            print(f"FAILED at cell {idx}: {res.error_in_exec}")
            raise res.error_in_exec
        if (idx + 1) % 5 == 0 or idx == len(code_cells) - 1:
            print(f"  Executed {idx + 1}/{len(code_cells)} cells successfully.")

    print("ALL NOTEBOOK CELLS EXECUTED SUCCESSFULLY IN-PROCESS!")

if __name__ == "__main__":
    test_notebook_execution()

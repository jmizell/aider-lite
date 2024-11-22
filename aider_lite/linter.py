import os
import re
import subprocess
import sys
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.simplefilter("ignore", category=FutureWarning)


class Linter:
    def __init__(self, encoding="utf-8", root=None):
        self.encoding = encoding
        self.root = root
        self.languages = dict(
            python=self.flake8_lint,
        )
        self.all_lint_cmd = None

    def set_linter(self, lang, cmd):
        if lang:
            self.languages[lang] = cmd
            return

        self.all_lint_cmd = cmd

    def get_rel_fname(self, fname):
        if self.root:
            try:
                return os.path.relpath(fname, self.root)
            except ValueError:
                return fname
        else:
            return fname

    def run_cmd(self, cmd, rel_fname, code):
        cmd += " " + rel_fname
        cmd = cmd.split()

        try:
            process = subprocess.Popen(
                cmd,
                cwd=self.root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding=self.encoding,
                errors="replace",
            )
        except OSError as err:
            print(f"Unable to execute lint command: {err}")
            return
        stdout, _ = process.communicate()
        errors = stdout
        if process.returncode == 0:
            return  # zero exit status

        cmd = " ".join(cmd)
        res = f"## Running: {cmd}\n\n"
        res += errors

        return self.errors_to_lint_result(rel_fname, res)

    def errors_to_lint_result(self, rel_fname, errors):
        if not errors:
            return

        linenums = []
        filenames_linenums = find_filenames_and_linenums(errors, [rel_fname])
        if filenames_linenums:
            filename, linenums = next(iter(filenames_linenums.items()))
            linenums = [num - 1 for num in linenums]

        return LintResult(text=errors, lines=linenums)

    def lint(self, fname, cmd=None):
        rel_fname = self.get_rel_fname(fname)
        try:
            code = Path(fname).read_text(encoding=self.encoding, errors="replace")
        except OSError as err:
            print(f"Unable to read {fname}: {err}")
            return

        if cmd:
            cmd = cmd.strip()
        if not cmd:
            # Simple extension based language detection
            ext = os.path.splitext(fname)[1].lower()
            lang = None
            if ext in ['.py']:
                lang = 'python'
            elif ext in ['.go']:
                lang = 'go'

            if not lang:
                return
            if self.all_lint_cmd:
                cmd = self.all_lint_cmd
            else:
                cmd = self.languages.get(lang)

        if callable(cmd):
            lintres = cmd(fname, rel_fname, code)
        elif cmd:
            lintres = self.run_cmd(cmd, rel_fname, code)
        else:
            return

        if not lintres:
            return

        res = "# Fix any errors below, if possible.\n\n"
        res += lintres.text
        res += "\n"
        res += basic_context(rel_fname, code, lintres.lines)

        return res

    def flake8_lint(self, rel_fname):
        fatal = "E9,F821,F823,F831,F406,F407,F701,F702,F704,F706"
        flake8_cmd = [
            sys.executable,
            "-m",
            "flake8",
            f"--select={fatal}",
            "--show-source",
            "--isolated",
            rel_fname,
        ]

        text = f"## Running: {' '.join(flake8_cmd)}\n\n"

        try:
            result = subprocess.run(
                flake8_cmd,
                cwd=self.root,
                capture_output=True,
                text=True,
                check=False,
                encoding=self.encoding,
                errors="replace",
            )
            errors = result.stdout + result.stderr
        except Exception as e:
            errors = f"Error running flake8: {str(e)}"

        if not errors:
            return

        text += errors
        return self.errors_to_lint_result(rel_fname, text)


@dataclass
class LintResult:
    text: str
    lines: list


def lint_python_compile(fname, code):
    try:
        compile(code, fname, "exec")  # USE TRACEBACK BELOW HERE
        return
    except Exception as err:
        end_lineno = getattr(err, "end_lineno", err.lineno)
        line_numbers = list(range(err.lineno - 1, end_lineno))

        tb_lines = traceback.format_exception(type(err), err, err.__traceback__)
        last_file_i = 0

        target = "# USE TRACEBACK"
        target += " BELOW HERE"
        for i in range(len(tb_lines)):
            if target in tb_lines[i]:
                last_file_i = i
                break

        tb_lines = tb_lines[:1] + tb_lines[last_file_i + 1:]

    res = "".join(tb_lines)
    return LintResult(text=res, lines=line_numbers)


def basic_context(fname, code, line_nums):
    """
    Basic context without tree-sitter
    """
    output = f"## See relevant lines below:\n\n"
    output += fname + ":\n"

    lines = code.splitlines()
    line_nums = set(line_nums)

    for i, line in enumerate(lines):
        if i in line_nums:
            output += f"{i + 1:4d} █ {line}\n"

    return output


def find_filenames_and_linenums(text, fnames):
    """
    Search text for all occurrences of <filename>:\\d+ and make a list of them
    where <filename> is one of the filenames in the list `fnames`.
    """
    pattern = re.compile(r"(\b(?:" + "|".join(re.escape(fname) for fname in fnames) + r"):\d+\b)")
    matches = pattern.findall(text)
    result = {}
    for match in matches:
        fname, linenum = match.rsplit(":", 1)
        if fname not in result:
            result[fname] = set()
        result[fname].add(int(linenum))
    return result


if __name__ == "__main__":

    def main():
        """
        Main function to parse files provided as command line arguments.
        """
        if len(sys.argv) < 2:
            print("Usage: python linter.py <file1> <file2> ...")
            sys.exit(1)

        linter = Linter(root=os.getcwd())
        for file_path in sys.argv[1:]:
            errors = linter.lint(file_path)
            if errors:
                print(errors)

    main()

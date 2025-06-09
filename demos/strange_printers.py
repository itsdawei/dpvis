"""664. Strange Printer

https://leetcode.com/problems/strange-printer/description/

There is a strange printer with the following two special properties:
- The printer can only print a sequence of the same character each time.
- At each turn, the printer can print new characters starting from and ending
  at any place and will cover the original existing characters.

Given a string s, return the minimum number of turns the printer needed to
print it.

Example 1:

Input: s = "aaabbb"
Output: 2
Explanation: Print "aaa" first and then print "bbb".

Example 2:

Input: s = "aba"
Output: 2
Explanation: Print "aaa" first and then print "b" from the second place of the
string, which will cover the existing character 'a'.

Constraints:
    1 <= s.length <= 100
    s consists of lowercase English letters.
"""

from dp import DPArray
from dp._visualizer import Visualizer


def strange_printer(s, OPT):
    n = len(s)

    # base cases
    OPT[:, :] = 0
    for i in range(n):
        OPT[i, i] = 1

    for left in range(n - 1, -1, -1):
        for right in range(left + 1, n):
            a = OPT[left + 1, right] + 1
            bs = []
            for i in range(left + 1, right + 1):
                if s[left] == s[i]:
                    # Delete s[l], ..., s[i-1] then s[i], ..., s[r].
                    b = OPT[left + 1, i - 1] + OPT[i, right]
                    bs.append(b)
            OPT[left, right] = min(bs + [a])
            OPT.annotate(f"Current string: {s[left:right+1]}")
            for i in range(n):
                for j in range(n):
                    OPT.annotate(s[i:j + 1], idx=(i, j))

    return OPT[0, n - 1]


s = "aaabbb"
OPT = DPArray((len(s), len(s)), array_name="Strange Printer")

strange_printer(s, OPT)

visualizer = Visualizer()
visualizer.add_array(OPT)

app = visualizer.create_app()
server = app.server

if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=True)

from dp import DPArray, display

"""
664. Strange Printer (https://leetcode.com/problems/strange-printer/description/)

There is a strange printer with the following two special properties:
- The printer can only print a sequence of the same character each time.
- At each turn, the printer can print new characters starting from and ending
  at any place and will cover the original existing characters.

Given a string s, return the minimum number of turns the printer needed to print it.

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


def strange_printer(s: str) -> int:
    n = len(s)
    OPT = DPArray((n, n), array_name="Strange Printer")

    # base cases
    OPT[:, :] = 0
    for i in range(n):
        OPT[i, i] = 1

    for l in range(n - 1, -1, -1):
        for r in range(l + 1, n):
            a = OPT[l + 1, r] + 1
            bs = []
            for i in range(l + 1, r + 1):
                if s[l] == s[i]:
                    # Delete s[l], ..., s[i-1] then s[i], ..., s[r].
                    b = OPT[l + 1, i - 1] + OPT[i, r]
                    bs.append(b)
            OPT[l, r] = min(bs + [a])
            OPT.annotate(f"Current string: {s[l:r+1]}")
            for i in range(n):
                for j in range(n):
                    OPT.annotate(s[i:j+1], idx=(i, j))

    display(OPT)
    return OPT[0, n - 1]


if __name__ == "__main__":
    s = "aaabbb"
    strange_printer(s)

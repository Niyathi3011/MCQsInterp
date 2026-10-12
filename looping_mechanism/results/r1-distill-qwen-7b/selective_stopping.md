# Selective stopping check: does re-stating the same answer predict it is right?

Natural loops. k = how many times the looping run states its committed answer and then doubts it
(over the whole 12k-token loop). correct = that committed answer is right.

| group | loops | correct overall | k=1 | k=2-3 | k=4-9 | k>=10 |
|---|---|---|---|---|---|---|
| nat_aqua | 28 | 14% | 25% (n=4) | 0% (n=6) | 22% (n=9) | 11% (n=9) |
| nat_logiqa | 95 | 38% | 35% (n=20) | 36% (n=14) | 41% (n=29) | 38% (n=32) |
| nat_math500 | 31 | 52% | 56% (n=9) | 55% (n=11) | 57% (n=7) | 25% (n=4) |
| all natural | 154 | 36% | 39% (n=33) | 35% (n=31) | 40% (n=45) | 31% (n=45) |

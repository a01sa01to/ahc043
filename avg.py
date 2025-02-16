import os

FILES = 100
s = []
ti = []
minScore = [0, 1e100]
maxScore = [0, 0]
minTime = [0, 1e100]
maxTime = [0, 0]

print("\033[1m\033[7m Calculating... \033[0m")

for i in range(1, FILES + 1):
    # with open(".{}tools{}res{}{}.txt".format(os.sep, os.sep, os.sep, i)) as f:
    with open(".{}tools{}err{}{}.txt".format(os.sep, os.sep, os.sep, i)) as f:
        flg = False
        lines = f.readlines()
        for line in lines:
            # if line.count("Score ="):
            if line.count("Cost ="):
                t = int(line.split(" = ")[1])
                flg = True
                if t:
                    s.append(t)
                    if t < minScore[1]:
                        minScore = [i, t]
                    if t > maxScore[1]:
                        maxScore = [i, t]
                    # result.write("{},{}\n".format(i, t))
                else:
                    print("Error on {}".format(i))
                    # result.write("{},{}\n".format(i, 0))
        print(f"No score on {i}") if not flg else None

    # with open(".{}tools{}err{}{}.txt".format(os.sep, os.sep, os.sep, i)) as f:
    #     lines = f.readlines()
    #     flg = False
    #     for line in lines:
    #         if line.count("Program end: "):
    #             flg = True
    #             t = int(line.split(" ")[2])
    #             ti.append(t)
    #             if(t < minTime[1]):
    #                 minTime = [i, t]
    #             if(t > maxTime[1]):
    #                 maxTime = [i, t]
    #     print(f"No time on {i}") if not flg else None

ave = sum(s) / len(s)
sd = (sum([(x - ave) ** 2 for x in s]) / len(s)) ** 0.5
print()
print("\033[1m\033[7m Results \033[0m")
print("\033[1mScore\033[0m")
print(f"  Average Score: {ave:.2f}")
print(f"  Minimum Score: {minScore[1]:.2f} on {minScore[0]}")
print(f"  Maximum Score: {maxScore[1]:.2f} on {maxScore[0]}")
print(f"  Standard Deviation: {sd:.2f}")
# print("\033[1mTime\033[0m")
# print(f"  Average Time: {(sum(ti) / len(ti)):.2f}ms")
# print(f"  Minimum Time: {minTime[1]:.2f}ms on {minTime[0]}")
# print(f"  Maximum Time: {maxTime[1]:.2f}ms on {maxTime[0]}")

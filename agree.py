import pandas as pd

def majority_vote(df_best, df1, df2, path):
    outputs = []
    filenames = df_best["File"].values

    for filename in filenames:
        if df1[df1["File"] == filename]["Label"].values == df2[df2["File"] == filename]["Label"].values:
            outputs.append(df1[df1["File"] == filename]["Label"].values.item())
        else:
            outputs.append(df_best[df_best["File"] == filename]["Label"].values.item())

    df_submission = pd.DataFrame({"File": filenames, "Label": outputs})
    df_submission.to_csv(path, index=False)



def check_agree(df1, df2):
    filenames = df1["File"].values
    ret = 0
    for filename in filenames:
        if df1[df1["File"] == filename]["Label"].values == df2[df2["File"] == filename]["Label"].values:
            ret += 1

    return ret


# print(agree)


df_best = pd.read_csv("data/submission_609.csv")
df_1 = pd.read_csv("data/submission_605.csv")
df_2 = pd.read_csv("data/submission_601.csv")
majority_vote(df_best, df_1, df_2, "data/submission_maj_vot.csv")

agree = check_agree(pd.read_csv("data/submission_609.csv"), pd.read_csv("data/submission_609_2.csv"))
print(agree)

agree = check_agree(pd.read_csv("data/submission_609.csv"), pd.read_csv("data/submission_maj_vot.csv"))
print(agree)

agree = check_agree(pd.read_csv("data/submission_605.csv"), pd.read_csv("data/submission_maj_vot.csv"))
print(agree)

agree = check_agree(pd.read_csv("data/submission_601.csv"), pd.read_csv("data/submission_maj_vot.csv"))
print(agree)


# agree = check_agree(pd.read_csv("data/submission_2.csv"), pd.read_csv("data/submission_2_tta_deeper.csv"))
# print(agree)
#
#
# agree = check_agree(pd.read_csv("data/submission_2.csv"), pd.read_csv("data/submission_deeper_cnn.csv"))
# print(agree)
#
# agree = check_agree(pd.read_csv("data/submission_2.csv"), pd.read_csv("data/submission.csv"))
# print(agree)

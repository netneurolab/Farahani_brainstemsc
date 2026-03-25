2022-11-15
From Simone Cauzzo (and Marta Bianciardi, Harvard)

20 compressed folders, one for each subject. In each folder, there are 66 files, 2 for each of the 33 medial and left labels of the Brainstem Navigator Atlas. These are .nii.gz files, for each label you can find:
- _coeff.nii.gz for correlation coefficients
- _pval.nii.gz for related p-values

In addition, in the folder you will find 3 text files with
- original acronnyms
- visualization acronyms
- full names
all ordered as in our published 2D functional connectomes.

2022-12-13
From Simone Cauzzo

ROI-based FC matrices, variables in .mat file refer to:
C_BSwithHO - Pearsons coefficients for each subject, in a #seeds x #seeds x #subjects matrix
C_BSwithHO_std - standard deviation of coefficients across subjects, in a #seeds x #seeds matrix
connectom_final - [-log10(p-values)] matrix, thresholded with bonferroni at p_value=0.0005, in a #seeds x #seeds matrix
labels_y - list of labels for the matrix columns
pCI_diagfix - p values for the group level t-test, with diagonal set to 1, in a #seeds x #seeds matrix
run_sN - just ignore
C_BSwithHO_mean - average of Pearsons coefficients across subjects, in a #seeds x #seeds matrix
C_Fischer - atanh(C_BSwithHO) (Fischer transform)
labels_x - list of labels for the matrix rows, in this case, same as labels_y
pCI - p-values for the group-level t-test, in a #seeds x #seeds matrix
p_BSwithHO - p-values for correlation at subject level
tCI - t-statistic on fischer coefficients
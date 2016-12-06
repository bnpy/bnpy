Q = load('ViterbiTestInput.mat');
[zHat, ~, ~] = hmmViterbiC(Q.logPiInit, Q.logPiTrans, Q.logEvidence');
save('ViterbiTestOutput.mat', 'zHat');

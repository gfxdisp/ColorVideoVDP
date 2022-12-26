% Find XYZ to CIE 2006 LMS approx. matrix

[lambda, XYZ] = load_spectra('ciexyz31.csv');
[lambda, LMS] = load_spectra('linss2_10e_1.csv');

% Weights used for our version of LMS
weights = [0.689903 0.348322 0.0371597];
LMS = LMS .* weights;

M=XYZ\LMS;
LMS_prime = XYZ*M;

clf;
subplot(1, 2, 1);
plot( lambda, XYZ );

subplot(1, 2, 2);
COLORs = { 'r', 'g', 'b' };
for cc=1:3
    plot( lambda, LMS(:,cc), 'Color', COLORs{cc} );
    hold on
    plot( lambda, LMS_prime(:,cc), '--', 'Color', COLORs{cc} );
end

M_t = M';


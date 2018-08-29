clear;
startup % execute gpml

ELL = 0.1; % length scale for of RBF kernel
N_TEST = 500; % number of test points
SIGMA_S = 0.5; % standard deviation of signal
SIGMA_D = 0.2; % standard deviation of noise on observation

rng('default');
rng(1);

% test function: sin(a*pi/(x+0.1)) with hyp optimization
N_VEC = [100, 500, 1000, 2000, 3000, 5000, 1e4, 2e4, 5e4, 1e5, 5e5]; % N training points
M_VEC = [10, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300]; % N inducing points

% number of points with acceptable time without hyp optimization 
full_n_thresh = 3e3;
fitc_n_thresh = 1e4;
kiss_n_thresh = 1e4;

ind_max = numel(N_VEC);

smse_full = NaN(ind_max, 1);
smse_fitc = NaN(ind_max, 1);
smse_kiss = NaN(ind_max, 1);
smse_lgg3 = NaN(ind_max, 1);
smse_lgg5 = NaN(ind_max, 1);

elapsed_full = NaN(ind_max, 1);
elapsed_fitc = NaN(ind_max, 1);
elapsed_kiss = NaN(ind_max, 1);
elapsed_lgg3 = NaN(ind_max, 1);
elapsed_lgg5 = NaN(ind_max, 1);

mu_full = NaN(ind_max, N_TEST);
mu_fitc = NaN(ind_max, N_TEST);
mu_kiss = NaN(ind_max, N_TEST);
mu_lgg3 = NaN(ind_max, N_TEST);
mu_lgg5 = NaN(ind_max, N_TEST);

X_master = rand(N_VEC(ind_max), 1);
Y_master = gen_y(X_master) + normrnd(0, SIGMA_D, N_VEC(ind_max), 1);

% generate test points
X_test = linspace(0, 1, N_TEST)';
Y_test = gen_y(X_test);

for ind = 1:ind_max
    disp('Number of data points:')
    disp(N_VEC(ind))

    % generate data
    X = X_master(1:N_VEC(ind), 1);
    Y = Y_master(1:N_VEC(ind), 1);
    Z = linspace(0, 1, M_VEC(ind))';

    % GP set up
    meanfunc = @meanConst;
    covfunc = @covSEiso;
    likfunc = @likGauss;
    hyp = struct('mean', [0], 'cov', [log(ELL), log(SIGMA_S)], 'lik', log(SIGMA_D));
    prior.cov = {{@priorGauss, log(ELL), 1e5}; {@priorDelta}};
    prior.lik = {{@priorDelta}};
    
    % Full GP training and prediction
    if N_VEC(ind) <= full_n_thresh
        inf = {@infPrior, @infGaussLik, prior};
        fprintf('Begin Full-GP.\n')
        tic;
        hyp_full = minimize(hyp, @gp, -20, inf, meanfunc, covfunc, likfunc, X, Y);
        [mu, s2] = gp(hyp_full, @infGaussLik, meanfunc, covfunc, likfunc, X, Y, X_test);
        elapsed_full(ind) = toc;
        smse_full(ind) = immse(Y_test, mu)/SIGMA_D^2;
        mu_full(ind, :) = mu';
        fprintf('Done Full gp.\n')
    end
    
    % FITC prediction (defined by s = 1)
    if N_VEC(ind) <= fitc_n_thresh
        u = Z;
        hyp.xu = u;
        covfuncF = {@apxSparse, {covfunc}, u};
        inf = {@infPrior, @(varargin) infGaussLik(varargin{:}, struct('s', 1.0)), prior};
        fprintf('Begin FITC-GP.\n')
        tic;
        hyp_fitc = minimize(hyp, @gp, -20, inf, meanfunc, covfuncF, likfunc, X, Y);
        [mF, s2F] = gp(hyp_fitc, inf, meanfunc, covfuncF, likfunc, X, Y, X_test);
        elapsed_fitc(ind) = toc;
        smse_fitc(ind) = immse(Y_test, mF)/SIGMA_D^2;
        mu_fitc(ind, :) = mF';
        fprintf('Done FITC.\n')
    end
    
    % KISS-GP setup
    meanfunc = {@meanConst};
    covfunc = {@covSEiso};
    likfunc = {@likGauss};
    hyp = struct('mean', [0], 'cov', [log(ELL), log(SIGMA_S)], 'lik', log(SIGMA_D));
    prior.mean = {{@priorDelta}};
    % KISS GP training and prediction
    if N_VEC(ind) <= kiss_n_thresh
        xg = Z;
        covg = {@apxGrid, {covfunc}, {xg}}; % grid prediction
        opt.cg_maxit = 1000; 
        opt.cg_tol = 2e-2; % was 1e-5
        opt.pred_var = 100;
        inf = {@infPrior, @(varargin) infGrid(varargin{:}, opt), prior};
        fprintf('Begin KISS-GP.\n')
        tic;
        hyp_kiss = minimize(hyp, @gp, -20, inf, meanfunc, covg, likfunc, X, Y);
        [postg, nlZg, dnlZg] = infGrid(hyp_kiss, meanfunc, covg, likfunc, X, Y, opt);
        [fmugf, fs2gf, ymugf, ys2gf] = postg.predict(X_test);
        elapsed_kiss(ind) = toc;
        smse_kiss(ind) = immse(Y_test, ymugf)/SIGMA_D^2;
        mu_kiss(ind, :) = ymugf';
        fprintf('Done KISS-GP.\n')
    end
    
    % LG-SWD-GP 3 band
    lg3hyp.ell = 0.6/M_VEC(ind);
    lg3hyp.sig_s = SIGMA_S;
    lg3hyp.sig_d = SIGMA_D;
    fprintf('Begin LG-GP 3 band.\n')
    tic;
    mulg3 = LGGP3b_1d(X, Y, Z, X_test, lg3hyp);
    elapsed_lgg3(ind) = toc;
    smse_lgg3(ind) = immse(Y_test, mulg3)/SIGMA_D^2;
    mu_lgg3(ind, :) = mulg3';
    fprintf('Done LG-GP 3 band.\n')

    % LG-SWD-GP 5 band
    lg5hyp.ell = 0.8/M_VEC(ind);
    lg5hyp.sig_s = SIGMA_S;
    lg5hyp.sig_d = SIGMA_D;
    fprintf('Begin LG-GP 5 band.\n')
    tic;
    mulg5 = LGGP5b_1d(X, Y, Z, X_test, lg5hyp);
    elapsed_lgg5(ind) = toc;
    smse_lgg5(ind) = immse(Y_test, mulg5)/SIGMA_D^2;
    mu_lgg5(ind, :) = mulg5';
    fprintf('Done LG-GP 5 band.\n')
    
end

save('lggp_exp_data.mat',...
'ELL', 'N_TEST', 'SIGMA_S', 'SIGMA_D', 'N_VEC', 'M_VEC',...
'hyp_full', 'hyp_fitc', 'hyp_kiss', 'lg3hyp', 'lg5hyp',...
'smse_full', 'smse_fitc', 'smse_kiss', 'smse_lgg3', 'smse_lgg5',...
'elapsed_full', 'elapsed_fitc', 'elapsed_kiss', 'elapsed_lgg3', 'elapsed_lgg5',...
'mu_full', 'mu_fitc', 'mu_kiss', 'mu_lgg3', 'mu_lgg5',...
'X_master', 'Y_master', 'X_test', 'Y_test');

%%
% function [y] = gen_y(x)
%     OMEGA = 42; % determines complexity of data, higher --> more complex
%     y = sin(OMEGA*pi*(x + 0.03)).*cos(OMEGA/3*pi*x);
% end
function [y] = gen_y(x)
    y = sin(5*pi./(x + 0.1));    
end
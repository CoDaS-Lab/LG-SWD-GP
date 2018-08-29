function mu = LLGGP5b_1d(x, y, u, xs, hyp)
% TODO: make this more efficient as is shown in the paper?
    ell = hyp.ell;
    sig_s = hyp.sig_s;
    sig_d = hyp.sig_d;
    n = numel(x);
    m = numel(u);

    % eigen decomposition
    B_uu = stack_eigvec(m);
    eigvals = stack_eigval(m, sig_s, ell, u(2) - u(1));

    % invert Lambda + sig^2*I (Snelson's notation)
    K_xu = covNN_5b(x, u, ell, sig_s);
    lambs = lambda(K_xu, B_uu, eigvals, sig_s);
    lambs_sig_inv = 1./(lambs + sig_d^2);
    L_xx = sparse(1:n, 1:n, lambs_sig_inv, n, n);

    % invert Q (Snelson's notation)
    Q_inv = approx_Qinv(K_xu, B_uu, L_xx, eigvals);
    
    % prediction
    K_su = covNN_5b(xs, u, ell, sig_s);
    mu = K_su*Q_inv*K_xu'*L_xx*y;
end

function Q_inv = approx_Qinv(K_xu, B_uu, L_xx, eigvals)
    [n, m] = size(K_xu);
    Q_inv = zeros(m);
    for i = 1:m
        cor = B_uu(:,i)'*K_xu'*L_xx*K_xu*B_uu(:,i);
        new_eigval = eigvals(i) + cor;
        Q_inv = Q_inv + (B_uu(:,i)*B_uu(:,i)')/new_eigval;
    end
end

function lambs = lambda(K_xu, B_uu, eigvals, sig)
    [n, m] = size(K_xu);
    lambs = zeros(n, 1);
    for i = 1:m
        lambs = lambs + (K_xu*B_uu(:,i)).^2/eigvals(i);
    end
    lambs = sig^2 - lambs; % for distance-kernel K_nn = sig^2 for all n
end

function K_eig = eigcomp(B_uu, eigvals)
    % test function
    m = numel(eigvals);
    K_eig = zeros(m);
    for i = 1:m
        K_eig = K_eig + eigvals(i)*(B_uu(:,i)*B_uu(:,i)'); % last product is outer product
    end
end

function B_uu = stack_eigvec(m)
    % eigvec stacked in cols
    B_uu = zeros(m);
    for i = 1:m
        B_uu(:,i) = kth_eigvec(i, m);
    end
end

function eigvals = stack_eigval(m, sig, ell, delta)
    eigvals = zeros(m, 1);
    for i = 1:m
        eigvals(i) = kth_eigenval(i, m, sig, ell, delta);
    end
end

function eigvec = kth_eigvec(k, m)
    theta = k*pi/(m + 1);
    phi = (2*m + 1)*theta;
    norm = (m + 1)/2;
    rind = (1:m)';
    eigvec = sin(rind*theta)/sqrt(norm);
end

function eigval = kth_eigenval(k, m, sig, ell, delta)
    theta = k*pi/(m + 1);
    alpha = exp(-0.5*(delta/ell)^2);
    eigval = sig^2*(1 + 2*alpha*cos(theta) + 2*alpha^4*cos(2*theta)); % 5 bands
end

function ind = findNN(val, vec)
    [~, ind] = min(abs(vec - val));
end

function k = rbf(x1, x2, ell, sig)
    k = sig^2*exp((-0.5)*((x1 - x2)/ell).^2);
end

function K = covNN_5b(x1, x2, ell, sig)
    % x2 meant to be in grid format
    % convention: column vectors as default
    
    n1 = numel(x1);
    n2 = numel(x2);
    
    x1array = (1:n1)'; % tmp index array for x1
    
    % 'midpoint': find x2(ind) closest to x1
    ind_mid = arrayfun(@(x_val) findNN(x_val, x2), x1);
    
    % contribiution from 'midpoint' to sparse matrix 
    s_i_mid = x1array;
    s_j_mid = ind_mid;
    s_i = s_i_mid;
    s_j = s_j_mid;
    
    % contribution from '1st upper point' to sparse matrix
    ind_upp1 = ind_mid + 1;
    j_upp1 = ind_upp1 > 0 & ind_upp1 <= n2; % use only points in range
    s_i = [s_i; x1array(j_upp1)];
    s_j = [s_j; ind_upp1(j_upp1)];
    
    % contribution from '1st lower point' to sparse matrix
    ind_low1 = ind_mid - 1;
    j_low1 = ind_low1 > 0 & ind_low1 <= n2;
    s_i = [s_i; x1array(j_low1)];
    s_j = [s_j; ind_low1(j_low1)];

    % contribution from '2nd upper point' to sparse matrix
    ind_upp2 = ind_mid + 2;
    j_upp2 = ind_upp2 > 0 & ind_upp2 <= n2; % use only points in range
    s_i = [s_i; x1array(j_upp2)];
    s_j = [s_j; ind_upp2(j_upp2)];
    
    % contribution from '2nd lower point' to sparse matrix
    ind_low2 = ind_mid - 2;
    j_low2 = ind_low2 > 0 & ind_low2 <= n2;
    s_i = [s_i; x1array(j_low2)];
    s_j = [s_j; ind_low2(j_low2)];

    % construct the values
    s_v_mid = rbf(x1, x2(ind_mid), ell, sig);
    s_v_upp1 = rbf(x1(j_upp1), x2(ind_upp1(j_upp1)), ell, sig);
    s_v_low1 = rbf(x1(j_low1), x2(ind_low1(j_low1)), ell, sig);
    s_v_upp2 = rbf(x1(j_upp2), x2(ind_upp2(j_upp2)), ell, sig);
    s_v_low2 = rbf(x1(j_low2), x2(ind_low2(j_low2)), ell, sig);

    % correct values
    kf = ind_mid == n2; % correction at the end
    s_v_mid(kf) = s_v_mid(kf) - rbf(x1(kf), x2(n2-2), ell, sig); % end correction
    ki = ind_mid == 1; % correction at the beginning
    s_v_mid(ki) = s_v_mid(ki) - rbf(x1(ki), x2(3), ell, sig); % end correction

    s_v = [s_v_mid; s_v_upp1; s_v_low1; s_v_upp2; s_v_low2];
    
    % construct the sparse matrix
    K = sparse(s_i, s_j, s_v, n1, n2);
end
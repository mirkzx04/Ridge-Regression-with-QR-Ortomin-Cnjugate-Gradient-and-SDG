function project()
    % Funzione di avvio del progetto
    % Chiama la funzione principale con i parametri di default

    % Parametri base
    n      = 500;   % numero campioni
    d      = 200;    % numero feature
    rng(1);         % seme random fisso per ripetibilità

    % Dati sintetici 
    X       = randn(n, d);
    true_w  = randn(d, 1);
    y       = X * true_w + 0.1 * randn(n, 1);

    % Sweep su lambda – valutiamo QR, CG, ORTOMIN, SGD
    lambdas = logspace(-4, 3, 20);
    m       = numel(lambdas);
    % Pre‑alloca metriche: righe = lambda, colonne = metodi
    tempi   = zeros(m,4);   % col1=QR col2=CG col3=ORT col4=SGD
    err_pesi= zeros(m,4);
    err_rmse= zeros(m,4);

    opts.lr = 1e-3; opts.epochs = 200; opts.shuffle = true;

    for k = 1:m
        lam = lambdas(k);

        % ---- QR ----
        [~, tempi(k,1), err_pesi(k,1), err_rmse(k,1)] = ...
            ridge_reg_with_qr(X, y, lam, true_w);

        % ---- CG ----
        tic;
        w_cg = ridge_reg_cg(X, y, lam, 1e-8, 1000);
        tempi(k,2)   = toc;
        err_pesi(k,2)= norm(w_cg-true_w);
        err_rmse(k,2)= sqrt(mean((X*w_cg - y).^2));

        % ---- ORTOMIN ----
        tic;
        w_ort = ridge_reg_ortomin(X, y, lam, 1e-8, 1000);
        tempi(k,3)   = toc;
        err_pesi(k,3)= norm(w_ort-true_w);
        err_rmse(k,3)= sqrt(mean((X*w_ort - y).^2));

        % ---- SGD ----
        tic;
        w_sgd = ridge_reg_sgd(X, y, lam, opts);
        tempi(k,4)   = toc;
        err_pesi(k,4)= norm(w_sgd-true_w);
        err_rmse(k,4)= sqrt(mean((X*w_sgd - y).^2));
    end


    % --------- Figura compatta con 3 metriche (loop per evitare linee mancanti) ----------
    figure('Name','Sweep lambda – tutte le metriche','Position',[100 100 900 700]);
    labels = {'QR','CG','ORT','SGD'};
    col    = lines(numel(labels));
    styles = {'-o','--d','-.s',':^'};   % QR CG ORT SGD  (ORT più visibile)
    drawOrder = [1 2 4 3];   % QR, CG, SGD, infine ORT

    % micro‑jitter per separare le curve sovrapposte (0.05% circa)
    jitter = 1 + [0, 5e-4, 1e-3, 1.5e-3];   % QR, CG, ORT, SGD

    % Tempo
    subplot(3,1,1); hold on; grid on
    for jj = 1:numel(drawOrder)
        j = drawOrder(jj);
        semilogx(lambdas .* jitter(j), tempi(:,j), styles{j}, ...
                 'Color', col(j,:), 'MarkerFaceColor', col(j,:), ...
                 'DisplayName', labels{j}, ...
                 'LineWidth', 2 * (j==2 || j==3) + 1.4 * (j==1 || j==4));
    end
    ylabel('Tempo [s]'); title('Tempo di risoluzione'); legend show

    % Errore pesi
    subplot(3,1,2); hold on; grid on
    for jj = 1:numel(drawOrder)
        j = drawOrder(jj);
        semilogx(lambdas .* jitter(j), err_pesi(:,j), styles{j}, ...
                 'Color', col(j,:), 'MarkerFaceColor', col(j,:), ...
                 'DisplayName', labels{j}, ...
                 'LineWidth', 2 * (j==2 || j==3) + 1.4 * (j==1 || j==4));
    end
    ylabel('\|w-w_{true}\|_2'); 
    set(gca,'YScale','log');
    title('Errore pesi stimati vs veri'); legend show

    % RMSE
    subplot(3,1,3); hold on; grid on
    for jj = 1:numel(drawOrder)
        j = drawOrder(jj);
        semilogx(lambdas .* jitter(j), err_rmse(:,j), styles{j}, ...
                 'Color', col(j,:), 'MarkerFaceColor', col(j,:), ...
                 'DisplayName', labels{j}, ...
                 'LineWidth', 2 * (j==2 || j==3) + 1.4 * (j==1 || j==4));
    end
    xlabel('\lambda'); ylabel('RMSE'); 
    set(gca,'YScale','log');
    title('RMSE predizione'); legend show

    % ------------------------------------------------------------------
    % Tempo vs lambda su range esteso (n fisso = 500, d=200)
    % ------------------------------------------------------------------
    lambdas_big = logspace(-20,3,24);
    tbig = zeros(numel(lambdas_big),4);
    for k = 1:numel(lambdas_big)
        lam = lambdas_big(k);
        tbig(k,1) = timed_qr(X, y, lam);
        tic; ridge_reg_cg(X, y, lam, 1e-8, 2000);  tbig(k,2)=toc;
        tic; ridge_reg_ortomin(X, y, lam, 1e-8, 2000); tbig(k,3)=toc;
        tic; ridge_reg_sgd(X, y, lam, opts);         tbig(k,4)=toc;
    end
    figure('Name','Tempo vs lambda (range esteso)','Position',[200 100 800 450]);
    hold on; grid on
    for j=1:numel(labels)
        semilogx(lambdas_big, tbig(:,j), '-o', 'Color', col(j,:), ...
                 'MarkerFaceColor', col(j,:), 'LineWidth',1.4, 'DisplayName', labels{j});
    end
    xlabel('\lambda'); ylabel('Tempo [s]');
    title('Tempo di esecuzione in funzione di \lambda (n=500, d=200)');
    legend show

    % ------------------------------------------------------------------
    % Tempo vs d (n fisso) per quattro valori di lambda
    % ------------------------------------------------------------------
    lam_list = [1e-4 1e-2 1 1e2];
    dims2 = [20 50 100 200 400 800];
    T_d = zeros(numel(dims2), numel(lam_list), 4);  % dims × lambda × metodo

    for di = 1:numel(dims2)
        dcur = dims2(di);
        Xd = randn(n, dcur);
        wtrue_d = randn(dcur,1);
        yd = Xd*wtrue_d + 0.1*randn(n,1);
        for li = 1:numel(lam_list)
            lam=lam_list(li);
            % QR
            T_d(di,li,1) = timed_qr(Xd,yd,lam);
            % CG
            tic; ridge_reg_cg(Xd,yd,lam,1e-8,2000);  T_d(di,li,2)=toc;
            % ORT
            tic; ridge_reg_ortomin(Xd,yd,lam,1e-8,2000); T_d(di,li,3)=toc;
            % SGD
            tic; ridge_reg_sgd(Xd,yd,lam,opts); T_d(di,li,4)=toc;
        end
    end

    % plot per ogni metodo un sottografo
    for meth = 1:4
        figure('Name',sprintf('Tempo vs d – %s', labels{meth}));
        hold on; grid on
        for li = 1:numel(lam_list)
            plot(dims2, T_d(:,li,meth), '-o', 'Color', col(li,:), ...
                 'MarkerFaceColor', col(li,:), 'LineWidth',1.4, ...
                 'DisplayName', sprintf('\\lambda=%.0e',lam_list(li)));
        end
        xlabel('d'); ylabel('Tempo [s]');
        title(sprintf('%s: tempo vs d (n=%d)', labels{meth}, n));
        legend show
    end

    %==============================================================================
    % Tempo di esecuzione in funzione della dimensione d (fisso n e lambda)
    %==============================================================================
    lambda0 = 1e-2;
    dims    = [20 50 100 200 400 800];
    nd      = numel(dims);
    time_d  = zeros(nd,4);  % QR CG ORT SGD

    for jd = 1:nd
        d_cur = dims(jd);
        Xd    = randn(n, d_cur);
        w_true_d = randn(d_cur,1);
        yd    = Xd*w_true_d + 0.1*randn(n,1);

        % QR
        [~, tQR, ~, ~] = ridge_reg_with_qr(Xd, yd, lambda0, w_true_d); 
        time_d(jd,1) = tQR;

        % CG
        tic; ridge_reg_cg(Xd, yd, lambda0, 1e-8, 2000); time_d(jd,2)=toc;

        % ORT
        tic; ridge_reg_ortomin(Xd, yd, lambda0, 1e-8, 2000); time_d(jd,3)=toc;

        % SGD
        tic; ridge_reg_sgd(Xd, yd, lambda0, opts); time_d(jd,4)=toc;
    end

    % Plot tempi vs dimensione
    figure('Name','Tempo vs dimensione d','Position',[150 150 700 450]);
    hold on; grid on
    for j = 1:numel(labels)
        plot(dims, time_d(:,j), '-o', 'Color', col(j,:), ...
             'MarkerFaceColor', col(j,:), 'LineWidth',1.4, 'DisplayName', labels{j});
    end
    xlabel('d (numero feature)'); ylabel('Tempo [s]');
    title(sprintf('Tempo di esecuzione (n=%d, \\lambda = %.0e)', n, lambda0));
    legend show
end


% Ridge Regression con QR
function [w_ridge, tempo_risoluzione, err_pesi, rmse] = ...
            ridge_reg_with_qr(X, y, lambda, true_w)
    % Dimensioni 
    [~, d] = size(X);

    % Matr. estesa
    Xtilde = [X; sqrt(lambda)*eye(d)];
    ytilde = [y; zeros(d,1)];

    % QR + soluzione
    tic;
    [Q, R]  = qr(Xtilde, 0);        % QR compatta
    w_ridge = R \ (Q' * ytilde);    % back-substitution
    tempo_risoluzione = toc;

    % Metriche
    err_pesi = norm(w_ridge - true_w);
    y_pred   = X * w_ridge;
    rmse     = sqrt(mean((y_pred - y).^2));

    % Log 
    % fprintf('λ = %.2e | t = %.4fs | ‖Δw‖ = %.2e | RMSE = %.2e\n', ...
    %         lambda, tempo_risoluzione, err_pesi, rmse);
end



% Ridge Regression con SDG (mini‑batch size = 1)

function w_sgd = ridge_reg_sgd(X, y, lambda, opts)
    % opts.lr        : learning rate (default 1e-3)
    % opts.epochs    : epochs (default 100)
    % opts.shuffle   : shuffle data each epoch (default true)
    % returns:
    %   w_sgd   : pesi stimati
    
    if nargin < 4, opts = struct(); end
    if ~isfield(opts,'lr'),      opts.lr      = 1e-3; end
    if ~isfield(opts,'epochs'),  opts.epochs  = 100;  end
    if ~isfield(opts,'shuffle'), opts.shuffle = true; end
    
    [n, d]  = size(X);
    w_sgd   = zeros(d,1);           % init a zero
    
    for e = 1:opts.epochs
        if opts.shuffle
            idx = randperm(n);
            X = X(idx,:); y = y(idx);
        end
        
        for i = 1:n
            xi = X(i,:)';                       % sample (d x 1)
            yi = y(i);
            % gradiente del costo Ridge per singolo campione
            grad = (xi' * w_sgd - yi) * xi + lambda * w_sgd;
            w_sgd = w_sgd - opts.lr * grad;
        end
        
    end
end



% Ridge Regression via Metodo ORTOMIN
function w_ortomin = ridge_reg_ortomin(X, y, lambda, tol, itmax)
    % Risolve (X'X + λI)w = X'y con Ortomin(k=1)
    if nargin < 4, tol   = 1e-8; end
    if nargin < 5, itmax = 1000; end
    
    A = X.'*X + lambda*eye(size(X,2));
    b = X.'*y;
    
    w   = zeros(size(b));          % guess iniziale
    r   = b - A*w;                 % residuo
    p   = r;                       % direzione iniziale
    
    while norm(r) > tol && itmax > 0
        itmax = itmax - 1;
        Ap  = A*p;
        alpha = (r' * Ap) / (Ap' * Ap); % step Ortomin(k=1)
        w  = w + alpha * p;
        r_new = r - alpha * Ap;
        beta  = (r_new' * Ap) / (Ap' * Ap);
        p  = r_new + beta * p;
        r  = r_new;
    end
    w_ortomin = w;
end

% Ridge Regression via Gradiente Coniugato (CG) su matrice SPD
function w_cg = ridge_reg_cg(X, y, lambda, tol, itmax)
    % Risolve (X'X + λI)w = X'y con CG
    if nargin < 4, tol   = 1e-8; end
    if nargin < 5, itmax = 1000; end
    
    A = X.'*X + lambda*eye(size(X,2));
    b = X.'*y;
    
    w   = zeros(size(b));          % guess iniziale
    r   = b - A*w;                 % residuo iniziale
    p   = r;
    rs_old = r' * r;
    
    while sqrt(rs_old) > tol && itmax > 0
        itmax = itmax - 1;
        Ap = A*p;
        alpha = rs_old / (p' * Ap);
        w  = w + alpha * p;
        r  = r - alpha * Ap;
        rs_new = r' * r;
        if sqrt(rs_new) < tol, break; end
        p = r + (rs_new / rs_old) * p;
        rs_old = rs_new;
    end
    w_cg = w;
end
function t = timed_qr(X,y,lambda)
    [~,d] = size(X);
    Xtil = [X; sqrt(lambda)*eye(d)];
    ytil = [y; zeros(d,1)];
    tic;
    [Q,R] = qr(Xtil,0);
    R\(Q'*ytil);
    t = toc;
end
function [E, D, M, L] = varIsing4(H, J, betaAll, opt_params)
% ВХОД
% H — внешнее магнитное поле, матрица размера vS x hS;
% J — параметр модели, равен 1 или -1;
% betaAll — вектор значений параметра \beta (вектор-строка длины \beta_0);
% opt_params — (необязательный параметр) параметры оптимизационного процесса, структура со следующими полями:
% 'max_iter' — максимальное количество итераций, по умолчанию = 300;
% 'tol_crit' — необходимая точность по значению нижней границы, по умолчанию = 10^{-4};
% 'num_start' — количество различных начальных приближений, по умолчанию = 1;
% ВЫХОД
% E — мат.ожидание энергии на один спин \frac{1}{N}\mathbb{E}E для каждой температуры, вектор-строка длины \beta_0;
% D — стандартное отклонение энергии на один спин \frac{1}{N}\sqrt{\mathbb{D}E} для каждой температуры, вектор-строка длины \beta_0;
% M — средняя магнетизация \sqrt{\mathbb{E}\mu^2} для каждой температуры, вектор-строка длины \beta_0;
% L — нижние границы для логарифмов нормировочных констант для каждой температуры, вектор-строка длины \beta_0.

    [v, h] = size(H);
    N = v * h;
    beta0 = length(betaAll);
      
    tol_crit = 10e-4;
    max_iter = 300;
    num_start = 1;
    if nargin > 3
        if isfield(opt_params, 'max_iter')
            max_iter = opt_params.max_iter;
        end
        if isfield(opt_params, 'tol_crit')
            tol_crit = opt_params.tol_crit;
        end
        if isfield(opt_params, 'num_start')
            num_start = opt_params.num_start;
        end
    end

    nb_set = [(1:N)' - v, (1:N)' - 1, (1:N)' + 1, (1:N)' + v];
    mask_l = [zeros(v, 1); ones(v * (h - 1), 1)];
    mask_u = reshape([zeros(1, h); ones(v - 1, h)], N, 1);
    mask_d = reshape([ones(v - 1, h); zeros(1, h)], N, 1);
    mask_r = [ones(v * (h - 1), 1); zeros(v, 1)];
    mask = [mask_l, mask_u, mask_d, mask_r];

    ind_set = repmat(nb_set', beta0, 1) +  ...
        repmat(reshape(repmat(0:N:((beta0 - 1) * N), 4, 1), 4 * beta0, 1), 1, N);

    ind_set = ind_set .* repmat(mask', beta0, 1);    

    edge_vert = reshape(repmat((1:(v - 1))', 1, h) + repmat((0:(h - 1)) * v, v - 1, 1), ...
            (v - 1) * h, 1);
    edge_hor = (1:(N - v))';
    edge = [edge_vert, edge_vert + 1; ...
            edge_hor, edge_hor + v];
    
    H = reshape(H, N, 1);
    H_rep = repmat(H, 1, beta0);

    L_best = zeros(1, beta0);
    q_best = zeros(N, beta0);
    for start = 1:num_start
        
        q = repmat(double(rand(N, 1) < 0.6), 1, beta0);
        L = zeros(1, beta0);


        for iter = 1:max_iter
            for i = 1:N
                [~, ~, ind] = find(ind_set(:, i));
                ind = reshape(ind, length(ind) / beta0, beta0);
                q(i, :) = 1 ./ (1 + exp(- 2 .* betaAll .* (H(i) + ...
                    J * sum(2 .* q(ind) - 1, 1))));
            end

            L_new = betaAll .* (sum(repmat(H, 1, beta0) .* (2 * q - 1), 1) + ...
                J * sum((2 * q(edge(:, 1), :) - 1) .* (2 * q(edge(:, 2), :) - 1), 1));

            check = q .* log(q) + (1 - q) .* log(1 - q);
            check(isnan(check)) = 0;
            L_new = L_new - sum(check, 1);

            if all(abs(L - L_new) < tol_crit)
                break;
            end
            L = L_new;
        end

        %Ex_i
        Ex = 2 * q - 1;
        E = - (sum(H_rep .* Ex, 1) + ...
            J * sum(Ex(edge(:, 1), :) .* Ex(edge(:, 2), :), 1));

        pairs = [reshape(repmat(1:N, N, 1), N^2, 1), repmat((1:N)', N, 1)];
        pairs = pairs(pairs(:, 1) ~= pairs(:, 2), :);
        M = sqrt((sum(Ex(pairs(:, 1) , :).* Ex(pairs(:, 2), :), 1) + N) / (N ^ 2));

        D1 = sum(H_rep(pairs(:, 1), :) .* H_rep(pairs(:, 2), :) .* ...
            Ex(pairs(:, 1), :) .* Ex(pairs(:, 2), :), 1) + ...
            sum(H_rep .^ 2, 1);

        len = size(edge, 1);
        trin = [repmat(edge, N, 1), reshape(repmat(1:N, len, 1), N * len, 1)];
        trin = trin(trin(:, 1) ~= trin(:, 3) & trin(:, 2) ~= trin(:, 3), :);

        D2 =  sum(Ex(trin(:, 1), :) .* Ex(trin(:, 2), :) .* ...
                Ex(trin(:, 3), :) .*  H_rep(trin(:, 3), :), 1) + ...
                sum(H_rep(edge(:, 1), :) .* Ex(edge(:, 2), :) + ...
                    H_rep(edge(:, 2), :) .* Ex(edge(:, 1), :), 1);

        quad = [reshape(repmat(reshape(edge, 1, []), len, 1), [], 2), ...
            repmat(edge, len, 1)];
        mask_eq = ~((quad(:, 1) == quad(:, 3)) & (quad(:, 2) == quad(:, 4)));
        quad = quad(mask_eq, :);

        quad_ik = quad(quad(:, 1) == quad(:, 3), :);    
        quad_il = quad(quad(:, 1) == quad(:, 4), :);   
        quad_jk = quad(quad(:, 2) == quad(:, 3), :);  
        quad_jl = quad(quad(:, 2) == quad(:, 4), :); 

        D3 = sum(Ex(edge(:, 1), :) .* Ex(edge(:, 2), :), 1) .^ 2 - ...
                sum((Ex(edge(:, 1), :) .* Ex(edge(:, 2), :)) .^2, 1) + len  + ...
                sum(Ex(quad_ik(:, 2), :) .* Ex(quad_ik(:, 4), :) .* ...
                (1 - Ex(quad_ik(:, 1), :) .^ 2), 1) + ...
                sum(Ex(quad_il(:, 2), :) .* Ex(quad_il(:, 3), :) .* ...
                (1 - Ex(quad_il(:, 1), :) .^ 2), 1) + ...
                sum(Ex(quad_jk(:, 1), :) .* Ex(quad_jk(:, 4), :) .* ...
                (1 - Ex(quad_jk(:, 2), :) .^ 2), 1) + ...
                sum(Ex(quad_jl(:, 1), :) .* Ex(quad_jl(:, 3), :) .* ...
                (1 - Ex(quad_jl(:, 2), :) .^ 2), 1);

        tmp =  D1 + 2 * J * D2 + J^2 * D3;
        D = sqrt(tmp - E .^ 2) / N;
        E = E / N;
        mask = (L > L_best);
         E_best(mask) = E(mask);
         D_best(mask) = D(mask);
         M_best(mask) = M(mask);
         L_best(mask) = L(mask);
     end
     E = E_best;
     D = D_best;
     M = M_best;
     L = L_best;
end

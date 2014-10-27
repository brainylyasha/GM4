function [E, D, M, S] = gibbsIsing6(H, J, betaAll, num_iter)
% ВХОД
% H — внешнее магнитное поле, матрица размера vS x hS;
% J — параметр модели, равен 1 или -1;
% betaAll — вектор значений параметра \beta (вектор-строка длины \beta_0);
% num_iter — количество итераций схемы Гиббса;
% ВЫХОД
% E — значения мат.ожиданий энергии на один спин \frac{1}{N}\mathbb{E}E для каждой температуры, вектор-строка длины \beta_0;
% D — значения стандартных отклонений энергии на один спин \frac{1}{N}\sqrt{\mathbb{D}E} для каждой температуры, вектор-строка длины\beta_0;
% M — значения средней магнетизации на один спин \sqrt{\mathbb{E}\mu^2} для каждой температуры, вектор-строка длины \beta_0;
% S — примеры конфигураций X для всех температур, массив размера vS x hS x \beta_0.
    
    [v, h] = size(H);
    N = v * h;
    beta0 = length(betaAll);
    S = repmat(2 * randi(2, v, h) - 3, [1, 1, beta0]);

    energy = zeros(num_iter, beta0);
    magnetic = zeros(num_iter, beta0);
    nb_set = [(1:N)' - v, (1:N)' - 1, (1:N)' + 1, (1:N)' + v, ...
               (1:N)' - (v + 1), (1:N)' + (v + 1)];
    mask_l = [zeros(v, 1); ones(v * (h - 1), 1)];
    mask_u = reshape([zeros(1, h); ones(v - 1, h)], N, 1);
    mask_d = reshape([ones(v - 1, h); zeros(1, h)], N, 1);
    mask_r = [ones(v * (h - 1), 1); zeros(v, 1)];
    mask_lu = [zeros(v, 1); reshape([zeros(1, h - 1); ones(v - 1, h - 1)], N - v, 1)];
    mask_rd = [reshape([ones(v - 1, h - 1); zeros(1, h - 1)], N - v, 1); zeros(v, 1)];
    mask = [mask_l, mask_u, mask_d, mask_r, mask_lu, mask_rd];
    

    ind_set = repmat(nb_set', beta0, 1) +  ...
        repmat(reshape(repmat(0:N:((beta0 - 1) * N), 6, 1), 6 * beta0, 1), 1, N);

    ind_set = ind_set .* repmat(mask', beta0, 1);    
    

    for iter = 1:num_iter
%        iter
        for i = 1:N
            [~, ~, ind] = find(ind_set(:, i));
            tmp = reshape(S(ind), length(ind) / beta0, beta0);
            tmp = 2 * betaAll .* (sum(tmp) * J + H(i));
            p = 1 ./ (1 + exp(-tmp));
            sample = rand(1, beta0);
            mask = (sample > p);
            col = ceil(i / v);
            row = i - (col - 1) * v;
            S(row, col, :) = reshape((-1) .^ mask, 1, 1, beta0);
        end
        energy(iter, :) = reshape(- J * (sum(sum(S(:, 1:(end - 1), :) .* S(:, 2:end, :), 1), 2) + ...
                        sum(sum(S(1:(end - 1), :, :) .* S(2:end, :, :))) + ...
                        sum(sum(S(1:(end - 1), 1:(end - 1), :) .* S(2:end, 2:end, :)))) - ...
                        sum(sum(repmat(H, [1, 1, beta0]) .* S, 1), 2), 1, beta0);
        magnetic(iter, :) =  reshape(sum(sum(S, 1), 2) / N, 1, beta0);
    end
    begin = round(num_iter / 3);
    energy = energy(begin:end, :);
    magnetic = magnetic(begin:end, :);
    q = length(energy);
    E = mean(energy) / N;
    D = sqrt((sum(energy .^ 2, 1) - sum(energy, 1) .^ 2 / q) / q) / N;
    M = sqrt(mean(magnetic .^ 2));

end

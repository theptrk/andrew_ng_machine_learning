function X_rec = recoverData(Z, U, K)
%RECOVERDATA Recovers an approximation of the original data when using the 
%projected data
%   X_rec = RECOVERDATA(Z, U, K) recovers an approximation the 
%   original data that has been reduced to K dimensions. It returns the
%   approximate reconstruction in X_rec.
%

% You need to return the following variables correctly.
X_rec = zeros(size(Z, 1), size(U, 1));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the approximation of the data by projecting back
%               onto the original space using the top K eigenvectors in U.
%
%               For the i-th example Z(i,:), the (approximate)
%               recovered data for dimension j is given as follows:
%                    v = Z(i, :)';
%                    recovered_j = v' * U(j, 1:K)';
%
%               Notice that U(j, 1:K) is a row vector.
%               

for i = 1:size(X_rec,1)
    % projected_x is (1, K)
    projected_x = Z(i, :);

    % initialize x
    x = zeros(1, size(U,1));

    % iterate through dimensions to recover value
    for j = 1:size(x, 2)
        % Ufor_dimension = (1, K)
        Ufor_dimension = U(j, 1:K);

        % set the value for this dimension
        % (1, K) * (1, K)'
        % (1, K) * (K, 1) = (1, 1)
        recovered_for_dimension = projected_x * Ufor_dimension';
        x(j) = recovered_for_dimension;
    end 

    X_rec(i,:) = x;
end


% =============================================================

end

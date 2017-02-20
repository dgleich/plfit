function [V,T] = lanczos(A,v,k,varargin)
% LANCZOS Run the Lanczos process for k steps.
%
% T = lanczos(A,v,k) returns the (k+1,k) tridiagonal matrix T with the
% Lanczos coefficients.  This call does not store the Lanczos vectors
% internally (and therefore, is more efficient than the 'full' call).
%
% [T,R] = lanczos(A,v,k) returns the (k+1,k) tridiagonal matrix T along
% with two vectors in R necessary to restart the process.  This call does 
% not store the Lanczos vectors internally.
%
% [V,T] = lanczos(A,v,k,'full') returns the k+1 set of Lanczos vectors and in
% opposite order.  This call does store the Lanczos vectors internally.
%
% [T,R] = lanczos(A,v,k,T,R) continues a Lanczos factorization for k steps.
% [V,T] = lanczos(A,v,k,T,V) continues a Lanczos factorization for k steps.
% [V,T] = lanczos(A,v,k,T,V,'full') see note 2 below.
%
% Note 1: in the "extend the factorization" calls, the vector v must be
% correctly sized for A, it does not have to be the starting vector for the
% Lanczos process.
%
% Note 2: There is a slight ambiguity in the "update" the factorization
% call.  In particular, if we try to update the factorization after a
% single step [V,T] = lanczos(A,v,1,'full'); [V,T] = lanczos(A,v,1,T,V);
% then the algorithm will erronously detect the second call as [T,R] =
% lanczos(A,v,k,T,R).  This occurs because the file uses size(V,2) == 2 to
% detect the [T,R] input instead of the [T,V] input.  To clarify the
% ambiguity, just specify the 'full' on the end.
%
% The matrix A must be an input that is a valid singleton input to the
% matrixop class or a matrixop class.  See matrixop for more details.
%
% See also LANCZOS, LANCZOSUB, LANCZOSLB, ARNOLDI, BLANCZOS

%
% David Gleich
% 21 May 2006
%

%
% 11 November 2006
% Updated the code to use the matrixop library.
%
% 11 December 2006
% - Updated to use the matrixop library more efficiently and only in the
% cases when the native Matlab type will not suffice.
% - Added comment about BLANCZOS
%

% A must be an input that creates a matrix operator
if (isnumeric(A) || islogical(A))
    Aop = A;
else
    Aop = matrixop(A);
end

n = length(v);

fullmode = 0;
extend = 0;

% parse the arguments
if isempty(varargin)
    % fall through, just here for simplicity of code
elseif length(varargin) == 1
    % the only option they could have specified is mode...
    fullmode = 1;
elseif length(varargin) == 2
    extend = 1;
    T = varargin{1};
    R = varargin{2};
    if (size(R,2) > 2)
        V = R;
        fullmode = 1;
    end
elseif (length(varargin) == 3)
    if (~strcmpi(varargin{3},'full'))
        error('lanczos:invalidArgument',...
            'invalid third optional argument (it should be ''full'')');
    end
    extend = 1;
    T = varargin{1};
    V = varargin{2};
    fullmode = 1;
else
    error('lanczos:invalidArgument',...
            'Too many input arguments.');
end
    
if ~fullmode
    if (~extend)
        % in this case, we are starting the factorization
        v = v./norm(v);
        a = zeros(k,1);
        b = zeros(k,1);

        b1 = 1;
        v1 = 0;

        r = v;

        ii=0;
        start = 0;
    else
        % in this case, we are extending the factorization
        r = R(:,2);
        v1 = R(:,1);
        
        astart = diag(T);
        bstart = diag(T,-1);
        
        a = [astart; zeros(k,1)];
        b = [bstart; zeros(k,1)];
        
        ii = length(astart);
        b1 = bstart(end);
        start = length(astart);
    end
    
    while (ii < k+start)
        v = r/b1;
        ii=ii+1;
        %p = Af(v);
        p = Aop*v;
        a(ii) = v'*p;
        r = p - a(ii)*v - b1*v1;
        b(ii) = norm(r);
        b1 = b(ii);
        v1 = v;
    end;

    bend = [0; b(1:end-1)];
    V = spdiags([b a bend ], [-1 0 1], k+start+1,k+start);
    T = [v1 r];
else
    
    % this mode saves all the vectors
    if (~extend)
        v = v./norm(v);
        a = zeros(k,1);
        b = zeros(k,1);

        V = zeros(n,k+1);

        b1 = 1;
        v1 = 0;

        r = v;

        ii=0;
        
        start = 0;
    else
        Vstart = V;
        V = [Vstart zeros(n,k)];
        
        % extract all the information from the stored vectors
        r = Vstart(:,end);
        v1 = Vstart(:,end-1);
        
        astart = diag(T);
        bstart = diag(T,-1);
        
        a = [astart; zeros(k,1)];
        b = [bstart; zeros(k,1)];
        
        ii = length(astart);
        b1 = bstart(end);
        start = length(astart);
        
        % we scale this vector, and then unscale it next to simplify the
        % code... this is probably rather unstable though and I really
        % shouldn't do it.
        r = r*b1;
    end
    
    while (ii < k + start)
        v = r/b1;
        ii=ii+1;
        V(:,ii) = v;
        %p = Af(v);
        p = Aop*v;
        a(ii) = v'*p;
        r = p - a(ii)*v - b1*v1;
        b(ii) = norm(r);
        b1 = b(ii);
        v1 = v;
    end;
    
    V(:,end) = r/b1;
    
    bend = [0; b(1:end-1)];
    T = spdiags([b a bend], [-1 0 1], k+1+start,k+start);
end;
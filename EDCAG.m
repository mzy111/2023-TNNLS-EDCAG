% Motivation: Maximize sum of intra-class similarity between sample and anchor layer 
% P = F*(F'*F)^-0.5
% Q = G*(G'*G)^-0.5
% max trace(P'*B*Q)

function [labelnew,obj,F,G,converge,converge_g,converge_f,t_EDCAG,h] = EDCAG(B,c,ITER)
% B is constructed anchor graph n*m
% c is # true classes

if nargin < 3
    ITER = 20;      % Maxiter 20
end

[n,m] = size(B);

tOptFG = tic;
%% Initialize F and G
F = zeros(n,c);
G = zeros(m,c);
fl = round(rand(1,n)*(c-1))+1;
gl = round(rand(1,m)*(c-1))+1;

for i = 1:n
    F(i,fl(i)) = 1;   % O(n)
end
for i = 1:m
    G(i,gl(i)) = 1;   % O(m)
end

for k = 1:c
    fc = length(find(fl==k));   % O(1)
    P(:,k) = F(:,k)./sqrt(fc);  % O(n_k) 属于第k类的样本数量
    gc = length(find(gl==k));   % O(1)
    Q(:,k) = G(:,k)./sqrt(gc);  % O(m_k) 属于第k类的锚点数量
end   

% U = B*Q;
% obj_init = 0;
% for k = 1:c
%     obj_init = obj_init + U(:,k)'*sparse(P(:,k));
% end

obj_init = trace(P'*B*Q);

obj = zeros(ITER+1,1);
obj(1) = obj_init;

converge = true;

%% Optimization
for h = 1:ITER
    %% fix F, update G
    V = B'*P;                                  % O(mn)
    [G,gg,~,~,converge_g] = updateGG(V,G);     % O(mct) t<10
    for cc = 1:c                 
        Q(:,cc) = G(:,cc)./sqrt(gg(cc)+eps);   % O()
    end
    %% fix G, update F
    U = B*Q;                                   % O(nc)
    [F,ff,~,~,converge_f] = updateFF(U,F);     % O(nct) t<10
    for cc = 1:c
        P(:,cc) = F(:,cc)./sqrt(ff(cc)+eps);   % O(1)
    end

    %% Record Objs
    obj(h+1) = trace(P'*B*Q);

    %% Converge
    err_obj = obj(h+1)-obj(h);
    per_obj = err_obj/obj(h);
    if err_obj < 0
        converge = false;
    end
%     if h>2 && abs(err_obj)<1e-5
    if h>2 && per_obj<1e-5
        break;
    end
end
t_EDCAG = toc(tOptFG);   % running time without anchor generation and anchor graph construction

%% Clutering Result
for i = 1:n
    [~,labelnew(i)] = find(F(i,:)==1);
end
end

function [F,ff,obj_F,changed,converge_f] = updateFF(U,F) % O(nc)
%% Preliminary
[n,c] = size(F);
obj_F = zeros(11,1);           

ff = sum(F);                        % O(nc)
uf = sum(U.*F);                     % O(nc)

up = zeros(1,c);
for cc = 1:c                        % O(c)
    up(cc) = uf(cc)./sqrt(ff(cc));  % O(1)
end
obj_F(1) = sum(up);                 % objf

changed = zeros(10,1);
incre_F = zeros(1,c);
converge_f = true;
%% Update
for iterf = 1:10                    % O(nct) t<5
    converged = true;
    for i = 1:n
        ui = U(i,:);
        [~,id0] = find(F(i,:)==1);
        for k = 1:c                          % O(c)
            if k == id0
                incre_F(k) = uf(k)/sqrt(ff(k)+eps) - (uf(k) - ui(k))/sqrt(ff(k)-1+eps);
            else
                incre_F(k) = (uf(k)+ui(k))/sqrt(ff(k)+1+eps) - uf(k)/sqrt(ff(k)+eps);
            end
        end

        [~,id] = max(incre_F);
        if id~=id0                           % O(1)
            converged = false;               
            changed(iterf) = changed(iterf)+1;
            F(i,id0) = 0; F(i,id) = 1;
            ff(id0) = ff(id0) - 1;           % id0 from 1 to 0, number -1
            ff(id)  = ff(id) + 1;            % id from 0 to 1，number +1
            uf(id0) = uf(id0) - ui(id0);
            uf(id)  = uf(id) + ui(id);
        end
    end
    if converged
        break;
    end

    for cc = 1:c
        up(cc) = uf(cc)/sqrt(ff(cc)+eps);
    end
    obj_F(iterf+1) = sum(up);

    err_obj_f = obj_F(iterf+1)-obj_F(iterf);
    if err_obj_f < 0
        converge_f = false;
    end
end
end


function [G,gg,obj_G,changed,converge] = updateGG(V,G) % O(mc)

[m,c] = size(G);
obj_G = zeros(11,1);

gg = sum(G)+eps*ones(1,c);     % tr(GTG) O(mc)
gv = sum(V.*G);                % tr(VTG) O(mc)

qv = zeros(1,c);
for cc = 1:c                        % O(c)
    qv(cc) = gv(cc)./sqrt(gg(cc));  % O(1)
end
obj_G(1) = sum(qv);            % objg O(c)

changed = zeros(10,1);
incre_G = zeros(1,c);
converge = true;
%% Update
for iterg = 1:10               % O(mct) t<10
    converged = true;
    for i = 1:m                           % O(mc)
        vi = V(i,:);
        [~,id0] = find(G(i,:)==1);
        for k = 1:c                       % O(c)
            if k == id0
                incre_G(k) = gv(k)/sqrt(gg(k)+eps) - (gv(k) - vi(k))/sqrt(gg(k)-1+eps);
            else
                incre_G(k) = (gv(k)+vi(k))/sqrt(gg(k)+1+eps) - gv(k)/sqrt(gg(k)+eps);
            end
        end

        [~,id] = max(incre_G);
        %         [~,id] = max(incre_g);     % 该行对应样本更新后的归属类别 1*1
        if id~=id0
            converged = false;               % not converge
            changed(iterg) = changed(iterg)+1; % change record
            G(i,id0) = 0;G(i,id) = 1;
            gg(id0) = gg(id0) - 1;           % id0 from 1 to 0, number -1
            gg(id)  = gg(id) + 1;            % id from 0 to 1, number +1
            gv(id0) = gv(id0) - vi(id0);     % id0 from 1 to 0, update gv
            gv(id)  = gv(id) + vi(id);       % id from 0 to 1, update gv
        end
    end
    if converged                             % m anchors traversal, false continue, true break
        break;
    end

    %% Obj tr(VT*Q)
    for cc = 1:c
        qv(cc) = gv(cc)/sqrt(gg(cc)+eps);
    end
    obj_G(iterg+1) = sum(qv);

    err_obj_g = obj_G(iterg+1)-obj_G(iterg);
    if err_obj_g < 0
        converge = false;
    end
end

end

    

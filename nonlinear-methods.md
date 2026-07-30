# scikit-fem で扱う非線形有限要素法

scikit-fem は非線形ソルバーを隠蔽するライブラリではないが、現在の解を
quadrature point に補間して弱形式を再アセンブルできる。このため、残差
\(R(u)\) と接線行列 \(J(u)=\partial R/\partial u\) を自分で定義すれば、
標準的な非線形有限要素法を素直に実装できる。

このディレクトリの例は scikit-fem 12 系、NumPy、SciPy のみを使い、GUI
なしで実行できる。すべて単位区間または単位正方形を使い、最後に収束履歴と
簡単な自己検査を表示する。

## 非線形性の分類

| 種類 | 典型例 | 離散化後に必要なもの | 対応する例 |
|---|---|---|---|
| 係数・材料非線形 | \(k(u)\), 塑性、超弾性 | 現在値で構成した残差と接線 | `picard-nonlinear-diffusion/`, `j2-plasticity/`, `newton-hyperelasticity/`, `mixed-incompressible-hyperelasticity/` |
| 反応・ソース非線形 | \(u^3\), \(e^u\) | 反応項とその導関数 | `newton-reaction-diffusion/`, `continuation-bratu/` |
| 幾何非線形 | 有限ひずみ、大変形 | 変形勾配と一貫接線 | `newton-hyperelasticity/` |
| 境界非線形 | 放射 \(q\propto T^4\)、接触 | 境界残差と境界接線 | 本文の境界積分の節 |
| 時間依存非線形 | Allen--Cahn、非線形拡散 | 時間離散化後の各ステップの非線形問題 | `allen-cahn-semi-implicit/` |
| 不等式・非滑らか | 接触、障害問題、降伏 | active-set、半滑らか Newton、正則化 | `contact-rigid-obstacle/` |
| 境界非線形 | 放射、非線形Robin条件 | 境界残差と境界接線 | `nonlinear-heat-radiation/` |
| 勾配依存非線形 | \(p\)-Laplacian | 正則化した勾配係数と接線 | `p-laplacian/` |
| 移流非線形 | 定常Navier--Stokes | Picard/Newtonと速度--圧力混合形式 | `steady-navier-stokes/` |
| 時間発展非線形 | 完全陰的Allen--Cahn | 各時間ステップのNewton法 | `allen-cahn-fully-implicit/` |
| 保存型時間発展 | Cahn--Hilliard | 相場--化学ポテンシャル混合形式 | `cahn-hilliard-mixed/` |
| 指数反応非線形 | Poisson--Boltzmann | 減衰Newtonと電荷継続 | `nonlinear-poisson-boltzmann/` |
| 連成非線形 | 熱--変形 | staggered/monolithic比較 | `thermo-mechanical-coupling/` |
| 変分不等式 | 膜の障害問題 | penalty/PDAS/半滑らかNewton | `obstacle-problem/` |
| 経路追跡 | 座屈後解析 | 荷重係数を含む弧長法 | `arc-length-buckling/` |
| 破壊・損傷 | phase-field破壊 | 変位--損傷の交互最小化 | `phase-field-fracture/` |
| 摩擦・履歴 | Coulomb摩擦接触 | stick/slip return mapping | `frictional-contact/` |
| 複合非線形 | 弾塑性押込み | J2 return mappingと接触 | `elastoplastic-indentation/` |
| 荷重方向非線形 | follower load | 変形依存外力と外力接線 | `follower-load/` |
| 動的非線形 | 有限ひずみ振動 | Newmark法と時刻内Newton法 | `nonlinear-elastodynamics/` |
| 動的非滑らか | 落下衝突 | Newmark法と接触／離脱 | `dynamic-impact-contact/` |
| 数値減衰 | generalized-\(\alpha\)衝突 | 高周波スペクトル制御 | `generalized-alpha-impact/` |
| 移動領域 | ALEメッシュ移動 | 調和／擬似弾性拡張と要素品質監視 | `ale-mesh-motion/` |

## 基本パターン

未知係数ベクトルを `x`、`basis.interpolate(x)` で得る離散場を `uh` とする。
非線形問題 \(R(x)=0\) に対する Newton 法は次の形になる。

```python
uh = basis.interpolate(x)
r = asm(residual, basis, uh=uh)
J = asm(jacobian, basis, uh=uh)

free = basis.complement_dofs(dirichlet_dofs)
dx = np.zeros_like(x)
dx[free] = solve(J[free][:, free], -r[free])
x += alpha * dx
```

`residual` と `jacobian` は同じ符号規約で書く。Dirichlet 条件が非斉次なら、
反復中も拘束自由度を所定値に保つ。反力を求める場合は、収束後の**縮約前の**
残差を拘束自由度上で読む。

### Picard（逐次代入）法

非線形係数を前反復値で固定して線形化する。例えば

\[
-\nabla\cdot((1+u^2)\nabla u)=f
\]

では \(1+(u^k)^2\) を固定し \(u^{k+1}\) を解く。接線の導出が不要で頑健だが、
通常は一次収束であり、強い非線形には緩和
\(u^{k+1}\leftarrow(1-\omega)u^k+\omega\tilde u^{k+1}\) が必要になる。

### Newton 法

\[
J(u^k)\Delta u=-R(u^k),\qquad u^{k+1}=u^k+\alpha\Delta u
\]

を解く。一貫した接線なら解の近傍で二次収束する。初期値が悪い場合は、
残差が減るまで \(\alpha\) を半減する backtracking、荷重増分、継続法を使う。
有限差分

\[
\frac{R(u+\varepsilon p)-R(u)}{\varepsilon}\approx J(u)p
\]

による directional derivative test は接線実装の有力な検査になる。

### 準 Newton・Jacobian-free Newton--Krylov

大規模問題では Jacobian を毎回組み直さず、BFGS 系更新や Krylov 法を利用
できる。SciPy の `root(..., method="krylov")` 等へ残差関数を渡す方法もあるが、
Dirichlet 自由度の除去、前処理、疎行列性を明示的に管理する方が有限要素問題
では扱いやすいことが多い。

### 継続法・荷重増分

難しい目標問題を直接解かず、パラメータ \(\lambda\) を小刻みに変え、
直前の解を初期値にする。単純な parameter continuation は限界点で止まる。
座屈後経路や turning point を追跡するには、\(\lambda\) も未知数にして弧長
拘束を追加する pseudo-arclength 法を用いる。

### 時間発展

完全陰解法では各時刻に Newton 問題を解く。一方、非線形項の一部を過去値で
評価する IMEX／半陰解法は各ステップが線形になり実装が簡単である。ただし、
時間刻み制約やエネルギー安定性は分割方法に依存する。

## 境界非線形

`FacetBasis` に対して境界残差と接線をアセンブルし、領域積分へ加える。
例えば放射境界

\[
-k\nabla T\cdot n=\epsilon\sigma(T^4-T_\infty^4)
\]

なら、境界残差は
\(\int_\Gamma\epsilon\sigma(T^4-T_\infty^4)v\,ds\)、接線は
\(\int_\Gamma4\epsilon\sigma T^3\,\delta T\,v\,ds\) である。

接触や障害問題は単なる滑らかな境界非線形ではない。摩擦なし接触では gap
\(g\)、圧縮接触力 \(\lambda\) に対して

\[
g\geq0,\qquad\lambda\geq0,\qquad g\lambda=0
\]

という相補条件を課す。代表的選択肢は
penalty（簡単だが penalty 依存）、Lagrange multiplier（鞍点問題）、
Nitsche、primal-dual active-set／半滑らか Newton である。接触面探索と
active set の更新が中心になる。

## ディレクトリ構成と可視化

各事例は独立したサブフォルダに置く。

```text
case-name/
├── main.py
└── result.png
```

`main.py` は解析、自己検査、Matplotlibによる可視化をまとめて実行し、
画像をスクリプトと同じ場所の `result.png` に保存する。対話GUIや実行時の
current working directoryには依存しない。

接線を持つ新しい事例では `common/nonlinear_verification.py` の
`directional_derivative_errors` を使い、中心差分による残差の方向微分と
\(J(u)d\) を比較する。有限差分刻みに対する誤差も結果画像へ含める。

## 各サンプル

### `ale-mesh-motion/`

上側の移動界面
\(d_y=-A(t)\sin^2(\pi x)\) を内部の流体メッシュへ伝えるALE
(Arbitrary Lagrangian--Eulerian) メッシュ移動を扱う。界面変位の調和拡張
\(-\Delta d_m=0\) と、メッシュを仮想的な線形弾性体とみなす擬似弾性拡張を
比較する。

ALE写像は \(\chi(X,t)=X+d_m(X,t)\)、メッシュ速度は
\(w_m=\partial\chi/\partial t\) である。流体の移流速度は物理速度そのもの
ではなく \(u_f-w_m\) になる。この例は流体方程式をまだ解かず、FSI計算の
前段となる変位・速度の生成と、要素Jacobian比および最小角による品質監視を
独立に検証する。最大振幅では調和拡張に局所的な要素反転が生じる一方、
擬似弾性拡張は正のJacobianを維持し、移動法の選択が計算可能範囲を左右する
ことも可視化する。

### `picard-nonlinear-diffusion/`

単位正方形で \(k(u)=1+u^2\) の非線形拡散を Picard 法で解く。既知の厳密解を
使った manufactured source により誤差も計算する。

### `newton-reaction-diffusion/`

単位正方形で
\(-\Delta u+u^3=f\) を一貫接線と backtracking 付き Newton 法で解く。
Picard が作りにくい一般的な残差ベース実装の最小例である。

### `continuation-bratu/`

単位正方形の Bratu 問題
\(-\Delta u-\lambda e^u=0\) を、\(\lambda\) の継続と減衰 Newton で解く。
指数非線形、多重解、初期値依存性を示す。ただし単純な parameter
continuation なので fold を越えてはいない。

### `allen-cahn-semi-implicit/`

単位正方形で
\(u_t-\epsilon^2\Delta u+u^3-u=0\) を解く。拡散と \(u^3\) を陰的な線形化、
\(-u\) を陽的に扱い、同じ質量・剛性行列を再利用する。

### `newton-hyperelasticity/`

単位正方形の圧縮性 neo-Hookean 体を変位制御で引張る。ベクトル P1 要素、
有限変形の第一 Piola 応力、四階一貫接線、荷重増分、backtracking Newton を
まとめた例である。

### `mixed-incompressible-hyperelasticity/`

ほぼ非圧縮neo-Hookean材料を、変位 \(u\) と圧力 \(p\) を独立未知数にした
混合形式で解く。変位にはベクトル \(P_2\)、圧力には \(P_1\) のTaylor--Hood
対を使い、体積ロッキングを避ける。採用した汎関数の材料部分は

\[
\Psi(F,p)=\frac{\mu}{2}(F:F-d)-\mu\ln J+p(J-1)-\frac{p^2}{2\kappa}
\]

であり、圧力方程式から \(J-1-p/\kappa=0\) を得る。
\(\kappa/\mu=10^4\) としても圧力を独立に近似するため、変位だけの低次要素に
比べて非圧縮制約を安定して表現できる。

Newton系は \(K_{uu},K_{up},K_{pu},K_{pp}\) の鞍点型ブロック行列になる。
結果画像には変形メッシュ、圧力、\(\det F\)、荷重変位曲線を表示する。

### `j2-plasticity/`

単位正方形を変位制御で引張り、その後除荷する小ひずみ・平面ひずみ J2
弾塑性の例である。von Mises 降伏、線形等方硬化、radial return mapping を
用いる。塑性ひずみと累積塑性ひずみは quadrature point ごとに保持し、
各荷重ステップの大域 Newton 収束後に一度だけ commit する。

この短い例では return mapping 全体を局所的に中心差分して algorithmic
tangent を作る。解析的一貫接線より計算量は多いが、構成則と接線導出を分離
でき、接線の検証用 reference としても役立つ。実務規模では解析的一貫接線
または自動微分へ置き換える。

弾塑性で重要なのは trial state と committed state を混同しないことである。
Newton 途中の塑性変数を恒久更新すると、反復経路に依存する誤った解になる。

### `contact-rigid-obstacle/`

単位正方形の線形弾性体を、中央ほど初期 gap が小さい滑らかな剛体障害物へ
変位制御で押し付ける。物体の構成則は線形だが、接触／非接触領域が未知なので
全体問題は非線形である。

同じ節点接触モデルを penalty 法と primal-dual active-set (PDAS) 法で解く。
penalty 法は小さな貫入を許す代わりに実装が簡単で、penalty を大きくすると
条件数が悪化する。PDAS は active node の gap を厳密にゼロとし、接触反力が
負になった節点を解放する。ここでは形状が一致する node-to-rigid 接触であり、
一般の曲面間接触には射影、接触積分、slave/master または mortar の設計が
追加で必要になる。

### `nonlinear-heat-radiation/`

単位長方形で、温度依存熱伝導率 \(k(T)=1+T^2/2\)、一様発熱、右境界からの
\(T^4\) 放射を同時に扱う。左端は周囲温度に固定し、上下は断熱とする。
領域残差に加えて `FacetBasis` で放射残差を組み立て、一貫接線にも

\[
\int_{\Gamma_{\rm rad}}4\epsilon\sigma(T_{\rm abs})^3\,\delta T\,v\,ds
\]

を加える。結果画像は温度、放射流束、Newton収束、接線の方向微分検証を示す。

### `p-laplacian/`

単位正方形、斉次Dirichlet境界、一様ソースについて

\[
-\nabla\cdot\left[
(\varepsilon^2+|\nabla u|^2)^{(p-2)/2}\nabla u
\right]=1
\]

を解く。\(p<2\) では \(|\nabla u|=0\) で係数が特異になり、\(p>2\) では
退化するため、\(\varepsilon=10^{-3}\) で正則化する。接線には等方部分に加え、
勾配方向のdyadicな項が現れる。\(p=1.5,2,3,4\) の解を比較し、\(p=1.5,3\)
について共通directional derivative testを行う。

### `steady-navier-stokes/`

Reynolds数100の二次元lid-driven cavityを、速度 \(P_2\)・圧力 \(P_1\) の
Taylor--Hood要素で解く。非線形移流項

\[
(\boldsymbol{u}\cdot\nabla)\boldsymbol{u}
\]

を、移流速度だけ前反復値に固定するPicard（Oseen）法と、さらに
\((\delta\boldsymbol{u}\cdot\nabla)\boldsymbol{u}\) を含めるNewton法で
線形化する。同じStokes解を初期値にして、残差履歴と最終解を比較する。
圧力の定数不定性は一自由度を固定して除く。結果画像は速度、圧力、
Picard/Newton収束履歴、中心線速度を示す。

### `allen-cahn-fully-implicit/`

Allen--Cahn方程式

\[
u_t-\epsilon^2\Delta u+u^3-u=0
\]

をBackward Eulerで完全陰的に離散化し、各時刻の非線形問題を減衰Newton法で
解く。既存の半陰解法と同一の初期値、メッシュ、時間刻みで比較する。完全陰的
接線は質量・拡散に加えて \((3u^2-1)\delta u\) を含む。結果画像は両手法の
相場、差、自由エネルギー、Newton反復数、接線検証を示す。

### `cahn-hilliard-mixed/`

Cahn--Hilliard方程式を相場 \(c\) と化学ポテンシャル \(\mu_c\) の二つの
二階方程式へ分割する。

\[
\frac{c^{n+1}-c^n}{\Delta t}-\Delta\mu_c^{n+1}=0,\qquad
\mu_c^{n+1}=(c^{n+1})^3-c^{n+1}-\epsilon^2\Delta c^{n+1}.
\]

両変数に \(P_1\) 要素を使い、完全陰的な混合ブロックNewton法で解く。
no-flux自然境界により全質量を保存しながら自由エネルギーを減少させる。
結果画像は初期・最終相場、化学ポテンシャル、エネルギー、質量誤差、
Newton反復数、混合接線検証を示す。

### `nonlinear-poisson-boltzmann/`

正負の局在固定電荷を持つ単位正方形で

\[
-\Delta\phi+\kappa^2\sinh\phi=\rho
\]

を解く。`sinh` と `cosh` は安全範囲で評価し、line searchは試行電位が指数
関数の許容範囲を越えた場合に棄却する。ゼロ初期値から目標電荷を直接解く
Newton法と、電荷振幅を0.1から1.0まで増やす継続法を比較する。結果画像は
電位、移動イオン電荷、継続経路、Newton反復数、接線検証を示す。

### `thermo-mechanical-coupling/`

温度依存弾性率と熱膨張を持つ小ひずみ熱弾性に、体積ひずみ依存の熱伝導率を
加えた双方向定常連成問題を解く。構成則と熱伝導率は

\[
\sigma=s(T)\,\mathbb{C}_0:(\epsilon(u)-\alpha T I),\qquad
k(T,u)=(1+\beta T^2)\exp(\gamma\nabla\cdot u)
\]

とする。熱問題と力学問題を交互に解くstaggered法と、
\(K_{uu},K_{uT},K_{Tu},K_{TT}\) を同時に解くmonolithic Newton法を比較する。
結果画像は温度、変形、von Mises応力、収束履歴、両手法の差、全ブロック
接線検証を示す。

### `obstacle-problem/`

周囲を固定した膜の変位 \(u\) が、内部の剛体障害物 \(\psi\) を突き抜けない
問題を解く。

\[
\min_u\left\{\frac12\int_\Omega|\nabla u|^2\,dx-\int_\Omega fu\,dx\right\},
\qquad u\geq\psi.
\]

離散相補条件は gap \(g=u-\psi\) と障害物反力 \(\lambda\) に対して
\(g\geq0,\lambda\geq0,g\lambda=0\) となる。有限penaltyで微小貫入を許す方法、
接触節点を更新するprimal-dual active-set法、Fischer--Burmeister関数を使う
半滑らかNewton法を比較する。結果画像は障害物、膜、反力、中心断面、
penalty依存性、非線形収束を示す。

### `arc-length-buckling/`

初期不整を持つ細長い圧縮性neo-Hookean柱を、荷重係数 \(\lambda\) も未知数に
した弧長法で圧縮する。各増分で釣合い

\[
R(u,\lambda)=f_{\rm int}(u)-\lambda f_{\rm ref}=0
\]

に加え、

\[
\|\Delta u\|^2+\alpha^2(\Delta\lambda)^2=\Delta s^2
\]

を課し、bordered Newton系を解く。これにより、通常の荷重制御では接線剛性が
特異になる極限点近傍でも、変位と荷重を同時に補正して座屈後経路を追跡できる。
結果画像は変形形状、荷重--短縮経路、横たわみ、corrector反復数、超弾性接線
検証を示す。

### `phase-field-fracture/`

反平面せん断（Mode III）のAT2 phase-field破壊を、変位 \(u\) と損傷
\(d\in[0,1]\) の交互最小化で解く。エネルギーは

\[
\mathcal{E}(u,d)=
\int_\Omega \left[((1-d)^2+\eta)\frac{\mu}{2}|\nabla u|^2
+G_c\left(\frac{d^2}{2\ell}+\frac{\ell}{2}|\nabla d|^2\right)\right]dx
\]

とする。履歴場
\(H(x,t)=\max_{\tau\leq t}\mu|\nabla u(x,\tau)|^2/2\) と
\(d^{n+1}\geq d^n\) の射影で亀裂治癒を防ぐ。左端中央の初期亀裂から損傷が
進展する過程を変位制御で追跡する。結果画像は初期・最終損傷、変位、
荷重変位曲線、弾性・破壊エネルギー、亀裂長、交互反復数を示す。

### `frictional-contact/`

弾性ブロックを曲面状剛体障害物へ押し付けた後、上面を水平方向へ往復させる。
法線方向はpenalty接触、接線方向はtrial traction

\[
t_{\rm tr}=k_t(u_t-s^n)
\]

を計算し、\(|t_{\rm tr}|\leq\mu\lambda_n\) ならstick、超えれば
\(t=\mu\lambda_n\operatorname{sign}(t_{\rm tr})\) としてslipを更新する。
sliding時には法線変位が摩擦限界へ影響する非対称接線を使う。結果画像は
変形、法線・摩擦力、累積すべり、摩擦ヒステリシス、stick/slip節点数、
非線形反復数を示す。

### `elastoplastic-indentation/`

平面ひずみJ2弾塑性体へ曲面剛体圧子を押し込み、その後完全除荷する。
積分点ではvon Mises降伏、等方硬化、radial return mappingを行い、上面では
摩擦なしpenalty接触のactive nodeを更新する。塑性状態は各荷重ステップの
大域Newton収束後にだけcommitする。除荷後には累積塑性ひずみと残留くぼみが
残る。結果画像はピーク変形、塑性域、接触力分布、押込みヒステリシス、
接触幅・塑性履歴、連成反復数を示す。

### `follower-load/`

有限変形neo-Hookean cantileverの右端に、端面回転へ追随する荷重を加える。
方向固定のdead loadでは外力ベクトルは一定だが、follower loadでは

\[
t(u)=-P\,\frac{F(u)e_y}{|F(u)e_y|}
\]

となり、残差だけでなく外力の変位微分もNewton接線へ含める必要がある。
全接線は一般に非対称になる。同じ荷重履歴でdead/follower応答を比較し、
結果画像は変形、荷重変位曲線、先端軌跡、内力・外力を合わせた接線検証を示す。

### `nonlinear-elastodynamics/`

有限変形neo-Hookean cantileverへ半正弦パルスを加え、その後の自由振動を
Newmark平均加速度法で解く。各時刻では

\[
M a_{n+1}+f_{\rm int}(u_{n+1})=f_{\rm ext}(t_{n+1})
\]

をNewton法で満たし、有効接線
\(M/(\beta\Delta t^2)+K_{\rm tan}(u_{n+1})\) を使う。同じ質量行列と初期接線
による線形動解析も並走させる。結果画像は変形スナップショット、先端振動、
位相軌道、運動・ひずみエネルギー、仕事とのエネルギー収支、時刻内Newton
反復数を示す。

### `dynamic-impact-contact/`

有限変形neo-Hookeanブロックを重力下で落下させ、剛体床との衝突・離脱・反発を
Newmark平均加速度法で解く。床とのgapが負のときだけpenalty接触残差と接線を
加えるため、各時刻のNewton反復中にもactive contactが変化する。結果画像は
衝突形状、重心運動、接触力、最小gap、運動・ひずみ・重力・接触エネルギー、
時刻内反復数を示す。

### `generalized-alpha-impact/`

1次元有限ひずみ弾性棒の落下衝突を、Newmark平均加速度法と
generalized-\(\alpha\)法で比較する。指定した高周波スペクトル半径
\(\rho_\infty\) から

\[
\alpha_m=\frac{2\rho_\infty-1}{\rho_\infty+1},\qquad
\alpha_f=\frac{\rho_\infty}{\rho_\infty+1}
\]

を定める。\(\rho_\infty\) を小さくすると衝突で励起される非物理的な高周波を
強く減衰できる一方、アルゴリズムエネルギーも失われる。結果画像は反発運動、
接触力、位相軌道、エネルギー減衰、接触力スペクトル、Newton反復数を示す。

## 実務上の確認項目

1. 残差ノルムと増分ノルムの両方を監視する。
2. Newton 接線を directional derivative test で検証する。
3. メッシュ・時間刻み・荷重刻みに対する収束性を確認する。
4. quadrature order を非線形項に合わせる。指数関数や有限変形では既定次数
   だけで不十分なことがある。
5. Newton 反復中に NaN、負の Jacobian determinant、材料の定義域外を検出し、
   line search で不正な trial state を拒否する。
6. 反復回数が増えたとき、まず接線の誤り、拘束条件、スケーリング、初期値を
   疑い、その後に前処理や別アルゴリズムを検討する。

実行例:

```bash
python picard-nonlinear-diffusion/main.py
python newton-reaction-diffusion/main.py
python continuation-bratu/main.py
python allen-cahn-semi-implicit/main.py
python newton-hyperelasticity/main.py
python mixed-incompressible-hyperelasticity/main.py
python j2-plasticity/main.py
python contact-rigid-obstacle/main.py
python nonlinear-heat-radiation/main.py
python p-laplacian/main.py
python steady-navier-stokes/main.py
python allen-cahn-fully-implicit/main.py
python cahn-hilliard-mixed/main.py
python nonlinear-poisson-boltzmann/main.py
python thermo-mechanical-coupling/main.py
python obstacle-problem/main.py
python arc-length-buckling/main.py
python phase-field-fracture/main.py
python frictional-contact/main.py
python elastoplastic-indentation/main.py
python follower-load/main.py
python nonlinear-elastodynamics/main.py
python dynamic-impact-contact/main.py
python generalized-alpha-impact/main.py
python ale-mesh-motion/main.py
```

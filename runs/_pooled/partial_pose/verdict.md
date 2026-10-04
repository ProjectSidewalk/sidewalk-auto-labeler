# #116 verdict (pre-registered rule)

## height auto (decides)

- (i) paterson partial: median -0.149 m vs off, -0.171 m vs partial-shuffled; p90 -0.278 vs off, -0.426 vs partial-shuffled -> ok
- (i) paterson partial-loco: median -0.148 m vs off, -0.167 m vs partial-loco-shuffled; p90 -0.278 vs off, -0.427 vs partial-loco-shuffled -> ok
- (i) bend partial: median -0.087 m vs off, -0.120 m vs partial-shuffled; p90 -0.175 vs off, -0.263 vs partial-shuffled -> ok
- (i) bend partial-loco: median -0.101 m vs off, -0.146 m vs partial-loco-shuffled; p90 -0.182 vs off, -0.328 vs partial-loco-shuffled -> ok
- (i) gainesville partial: median -0.205 m vs off, -0.229 m vs partial-shuffled; p90 -0.354 vs off, -0.494 vs partial-shuffled -> ok
- (i) gainesville partial-loco: median -0.140 m vs off, -0.151 m vs partial-loco-shuffled; p90 -0.338 vs off, -0.283 vs partial-loco-shuffled -> ok
- (i) sao_paulo partial: median -0.116 m vs off, -0.120 m vs partial-shuffled; p90 -0.234 vs off, -0.254 vs partial-shuffled -> ok
- (i) sao_paulo partial-loco: median -0.129 m vs off, -0.153 m vs partial-loco-shuffled; p90 -0.238 vs off, -0.306 vs partial-loco-shuffled -> ok
- laurens_gsv: 300 common pairs < 500: not scored by (i)
- (ii/iii) paterson partial: off-pool R@2.5 +1.6 pt; unplaceable GT marks -3 (limit 9.2)
- (ii/iii) paterson partial-loco: off-pool R@2.5 +1.6 pt; unplaceable GT marks -2 (limit 9.2)
- (ii/iii) bend partial: off-pool R@2.5 -1.3 pt; unplaceable GT marks -1 (limit 7.9)
- (ii/iii) bend partial-loco: off-pool R@2.5 -1.3 pt; unplaceable GT marks -2 (limit 7.9)
- (ii/iii) gainesville partial: off-pool R@2.5 +2.0 pt; unplaceable GT marks -1 (limit 5.1)
- (ii/iii) gainesville partial-loco: off-pool R@2.5 -1.0 pt; unplaceable GT marks -1 (limit 5.1)
- (ii/iii) sao_paulo partial: off-pool R@2.5 +0.9 pt; unplaceable GT marks +0 (limit 5.5)
- (ii/iii) sao_paulo partial-loco: off-pool R@2.5 +0.9 pt; unplaceable GT marks +0 (limit 5.5)
- (ii/iii) laurens_gsv partial: off-pool R@2.5 +0.0 pt; unplaceable GT marks +0 (limit 5.2)
- (ii/iii) laurens_gsv partial-loco: off-pool R@2.5 -1.0 pt; unplaceable GT marks +0 (limit 5.2)
- (iv) bend partial: inventory median -0.021 m, p90 -0.052 m vs off (limit +0.10)
- (iv) bend partial-loco: inventory median -0.023 m, p90 -0.052 m vs off (limit +0.10)
- (iv) gainesville partial: inventory median -0.061 m, p90 -0.101 m vs off (limit +0.10)
- (iv) gainesville partial-loco: inventory median -0.038 m, p90 -0.071 m vs off (limit +0.10)
- clause (i): PASS
- clause (ii): FAIL -- bend/partial, bend/partial-loco
- clause (iii): PASS
- clause (iv): PASS
- VERDICT @ auto: FAIL

## height 2.6 (reported)

- (i) paterson partial: median -0.053 m vs off, -0.082 m vs partial-shuffled; p90 +0.001 vs off, -0.136 vs partial-shuffled -> FAIL
- (i) paterson partial-loco: median -0.059 m vs off, -0.095 m vs partial-loco-shuffled; p90 +0.003 vs off, -0.131 vs partial-loco-shuffled -> FAIL
- (i) bend partial: median -0.087 m vs off, -0.111 m vs partial-shuffled; p90 -0.135 vs off, -0.253 vs partial-shuffled -> ok
- (i) bend partial-loco: median -0.098 m vs off, -0.136 m vs partial-loco-shuffled; p90 -0.149 vs off, -0.322 vs partial-loco-shuffled -> ok
- (i) gainesville partial: median -0.052 m vs off, -0.083 m vs partial-shuffled; p90 -0.017 vs off, -0.093 vs partial-shuffled -> ok
- (i) gainesville partial-loco: median -0.060 m vs off, -0.062 m vs partial-loco-shuffled; p90 -0.099 vs off, -0.050 vs partial-loco-shuffled -> ok
- (i) sao_paulo partial: median -0.125 m vs off, -0.132 m vs partial-shuffled; p90 -0.212 vs off, -0.297 vs partial-shuffled -> ok
- (i) sao_paulo partial-loco: median -0.126 m vs off, -0.147 m vs partial-loco-shuffled; p90 -0.193 vs off, -0.268 vs partial-loco-shuffled -> ok
- laurens_gsv: 302 common pairs < 500: not scored by (i)
- (ii/iii) paterson partial: off-pool R@2.5 +0.6 pt; unplaceable GT marks -1 (limit 8.8)
- (ii/iii) paterson partial-loco: off-pool R@2.5 +0.6 pt; unplaceable GT marks -1 (limit 8.8)
- (ii/iii) bend partial: off-pool R@2.5 +1.9 pt; unplaceable GT marks -1 (limit 7.8)
- (ii/iii) bend partial-loco: off-pool R@2.5 +1.3 pt; unplaceable GT marks +0 (limit 7.8)
- (ii/iii) gainesville partial: off-pool R@2.5 -6.2 pt; unplaceable GT marks +4 (limit 4.9)
- (ii/iii) gainesville partial-loco: off-pool R@2.5 -3.1 pt; unplaceable GT marks +3 (limit 4.9)
- (ii/iii) sao_paulo partial: off-pool R@2.5 -0.9 pt; unplaceable GT marks +0 (limit 5.4)
- (ii/iii) sao_paulo partial-loco: off-pool R@2.5 -0.9 pt; unplaceable GT marks +0 (limit 5.4)
- (ii/iii) laurens_gsv partial: off-pool R@2.5 +2.9 pt; unplaceable GT marks +0 (limit 5.2)
- (ii/iii) laurens_gsv partial-loco: off-pool R@2.5 +1.9 pt; unplaceable GT marks +0 (limit 5.2)
- (iv) bend partial: inventory median -0.024 m, p90 -0.048 m vs off (limit +0.10)
- (iv) bend partial-loco: inventory median -0.025 m, p90 -0.047 m vs off (limit +0.10)
- (iv) gainesville partial: inventory median -0.062 m, p90 -0.079 m vs off (limit +0.10)
- (iv) gainesville partial-loco: inventory median -0.029 m, p90 -0.068 m vs off (limit +0.10)
- clause (i): FAIL -- paterson/partial, paterson/partial-loco
- clause (ii): FAIL -- gainesville/partial, gainesville/partial-loco
- clause (iii): PASS
- clause (iv): PASS
- VERDICT @ 2.6: FAIL

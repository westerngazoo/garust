# Filosofía de diseño — garust

## Principios

- Zero dependencies by default; `no_std` + allocation-free hot paths.
- Compile-time Cayley tables over runtime dispatch.
- Motors for composition; matrices only where bulk throughput wins.
- Honest benchmarking — document tradeoffs vs matrix libraries.

## Contexto

From-scratch zero-dependency Clifford algebra library with PGA/CGA geometry, rigid-body physics, and motor-native animation.

## Trade-offs explícitos

Este proyecto prioriza coherencia con los principios anteriores sobre conveniencia ad-hoc.
Cuando una decisión contradice un principio, debe documentarse como ADR o RFC.

## Relación con el ecosistema

- **motoreel**
- **guion**
- **ufl**
- **physics-lab**
- **VIGA**
- **goose-rover**

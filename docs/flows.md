# Flujos — garust

## Flujo principal

```mermaid
flowchart LR
MV[Multivector Cl P,Q,R] --> GP[Geometric Product]
GP --> ROT[Rotors / Motors]
ROT --> GEO[PGA incidence]
ROT --> PHY[Physics integration]
PHY --> ANIM[Motor tracks]
```

## Descripción paso a paso

1. **Entry** — `src/lib.rs` re-exports workspace crates; pick alias (`Pga3`, `Vga3`, `Cga3`).
1. **Core algebra** — `garust-core`: Cayley tables at compile time → wedge/inner/geometric product.
1. **Geometry** — `garust-geo`: planes wedge to points, motors compose rigid motion via screw slerp.
1. **Physics** — `garust-physics`: integrate rigid bodies → contacts → state for anim/recording.
1. **Animation** — `garust-anim`: record motor tracks consumed by motoreel downstream.

## Diagrama PlantUML

Equivalente PlantUML del flujo principal (misma topología que el diagrama Mermaid):

```plantuml
@startuml
title garust — flujo principal
note as N1
Ver flows.md Mermaid para detalle;
exportar con herramientas mermaid→plantuml si se prefiere editar en PlantUML.
end note
@enduml
```

## Estados y casos borde

Consulta los tests de integración y los RFC/requirements del proyecto para flujos de error y recuperación.

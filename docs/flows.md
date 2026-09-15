# Flujos — garust

## Flujo principal (Mermaid)

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
2. **Core algebra** — `garust-core`: Cayley tables at compile time → wedge/inner/geometric product.
3. **Geometry** — `garust-geo`: planes wedge to points, motors compose rigid motion via screw slerp.
4. **Physics** — `garust-physics`: integrate rigid bodies → contacts → state for anim/recording.
5. **Animation** — `garust-anim`: record motor tracks consumed by motoreel downstream.

## Secuencia (PlantUML)

Fuente: [`diagrams/flow-sequence.puml`](./diagrams/flow-sequence.puml)

```plantuml
@startuml
title garust — secuencia principal

participant "Entry" as Entry0
participant "Core algebra" as Corealgebra1
participant "Geometry" as Geometry2
participant "Physics" as Physics3
participant "Animation" as Animation4

Entry0 -> Corealgebra1: `garust-core`: Cayley tables at compile time → wedge/inner/geometric product.
Corealgebra1 -> Geometry2: `garust-geo`: planes wedge to points, motors compose rigid motion via screw slerp.
Geometry2 -> Physics3: `garust-physics`: integrate rigid bodies → contacts → state for anim/recording.
Physics3 -> Animation4: `garust-anim`: record motor tracks consumed by motoreel downstream.

@enduml
```

## Componentes / estados (PlantUML)

Fuente: [`diagrams/flow-architecture.puml`](./diagrams/flow-architecture.puml)

```plantuml
@startuml
title garust — flujo de componentes
start
:MVMultivector;
:GPGeometric;
:GP;
:ROTRotors;
:ROT;
:GEOPGA;
:PHYPhysics;
:PHY;
:ANIMMotor;
stop

@enduml
```

## Estados y casos borde

Consulta los tests de integración y los RFC/requirements del proyecto para flujos de error y recuperación.

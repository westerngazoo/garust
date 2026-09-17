# Recorrido del código — garust

Guía orientada a desarrolladores para entender dónde vive cada responsabilidad.

### 1. Entry

`src/lib.rs` re-exports workspace crates; pick alias (`Pga3`, `Vga3`, `Cga3`).

### 2. Core algebra

`garust-core`: Cayley tables at compile time → wedge/inner/geometric product.

### 3. Geometry

`garust-geo`: planes wedge to points, motors compose rigid motion via screw slerp.

### 4. Physics

`garust-physics`: integrate rigid bodies → contacts → state for anim/recording.

### 5. Animation

`garust-anim`: record motor tracks consumed by motoreel downstream.

## Punto de entrada recomendado

Empieza por el README del proyecto y el módulo/crate principal listado en la documentación de arquitectura.

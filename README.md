# MCP Atlassian

Este proyecto es una integración personalizada que desarrollé para conectar productos de Atlassian (Confluence y Jira) con asistentes de inteligencia artificial y flujos de trabajo automatizados. Está pensado para facilitar la gestión y consulta de información en entornos empresariales que utilizan Atlassian, tanto en la nube como en instalaciones propias (Server/Data Center).

## ¿Qué puedes hacer con MCP Atlassian?

- **Actualizaciones automáticas en Jira**: Puedes pedirle a tu asistente que actualice tickets de Jira a partir de notas de reuniones o resúmenes.
- **Búsqueda inteligente en Confluence**: Encuentra documentos clave en Confluence y obtén resúmenes automáticos.
- **Filtrado avanzado de incidencias**: Consulta rápidamente los bugs urgentes o tareas prioritarias en tus proyectos Jira.
- **Creación y gestión de contenido**: Automatiza la creación de documentos técnicos o páginas de diseño en Confluence.

## Compatibilidad

| Producto        | Tipo de despliegue        | Estado de soporte           |
|----------------|--------------------------|-----------------------------|
| **Confluence** | Cloud                    | ✅ Totalmente soportado      |
| **Confluence** | Server/Data Center       | ✅ Soportado (versión 6.0+)  |
| **Jira**       | Cloud                    | ✅ Totalmente soportado      |
| **Jira**       | Server/Data Center       | ✅ Soportado (versión 8.14+) |

## Autenticación

El sistema soporta autenticación mediante tokens de API para Atlassian Cloud y tokens de acceso personal para Server/Data Center. Es importante mantener estos tokens seguros y no compartirlos.

## Principales herramientas disponibles

### Para Jira
- Obtener detalles de incidencias
- Buscar incidencias usando JQL
- Crear, actualizar y eliminar incidencias
- Transicionar estados de tickets
- Añadir comentarios y worklogs
- Gestión de sprints y tableros ágiles

### Para Confluence
- Buscar contenido usando CQL
- Obtener y actualizar páginas
- Crear nuevas páginas y etiquetas
- Gestionar comentarios

## Seguridad

- Nunca compartas tus tokens de API ni tus archivos de configuración sensibles.
- Mantén tus credenciales en un lugar seguro y privado.
- Si tienes dudas sobre buenas prácticas de seguridad, no dudes en consultarme.

## Contribuciones

Estoy abierto a sugerencias y mejoras. Si deseas contribuir, puedes contactarme directamente para coordinar cambios o nuevas funcionalidades.

## Licencia

Este proyecto está bajo licencia MIT. No es un producto oficial de Atlassian.

---

*Modificado y mantenido por  Gabriel M.*

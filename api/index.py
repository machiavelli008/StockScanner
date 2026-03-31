# Самый минимальный ASGI app для Vercel
async def app(scope, receive, send):
    """ASGI application - минимальный функционирующий app для Vercel"""
    if scope['type'] != 'http':
        return
    
    path = scope['path']
    method = scope['method']
    
    # Health check
    if path == '/health' and method == 'GET':
        await send({
            'type': 'http.response.start',
            'status': 200,
            'headers': [[b'content-type', b'application/json']],
        })
        await send({
            'type': 'http.response.body',
            'body': b'{"status":"ok"}',
        })
        return
    
    # Default 404
    await send({
        'type': 'http.response.start',
        'status': 200,
        'headers': [[b'content-type', b'application/json']],
    })
    await send({
        'type': 'http.response.body',
        'body': b'{"message":"StockScanner API running","health":true}',
    })






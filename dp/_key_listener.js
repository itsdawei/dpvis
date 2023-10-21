document.addEventListener('DOMContentLoaded', function(){
    document.focus(); 

    document.addEventListener('keydown', function(event) {
        let key = event.keyCode;
        if (key === 37 || key === 39) {
            let store_keypress = document.getElementById('store-keypress');
            store_keypress.setAttribute('data-dash-data', JSON.stringify(key));
            store_keypress.dispatchEvent(new Event('change'));
        }
    });
});

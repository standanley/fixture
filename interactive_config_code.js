var pin_name_inc = 0;


function add_pin() {
    const pin_container = document.getElementById('pin_container');
    const pin_div = document.createElement('div')
    const pin_id = 'pin' + String(pin_name_inc);
    pin_name_inc++;
    pin_div.id = pin_id;
    pin_div.setAttribute('name', 'pin_div')
    pin_container.appendChild(pin_div);

    pin_div.appendChild(document.createElement('br'));

    //var name_div = document.createElement('div');
    //var name_text = document.createElement('label');
    //name_text.innerHTML = 'Pin name: ';
    //var name_box = document.createElement('input');
    //name_box.type = 'text';
    //name_box.id = 'pin_name';
    //name_div.appendChild(name_text);
    //name_div.appendChild(name_box);
    //pin_div.appendChild(name_div);
    const name_element = text('Pin name: ', pin_id + '_name');
    pin_div.appendChild(name_element);

    const electricaltype = radio(pin_id + 'electricaltype',
        {'voltage': [empty, empty],
         'current': [empty, empty]}
    );
    //pin_div.appendChild(label('Electrical type: ', electricaltype));
    pin_div.appendChild(electricaltype);


    const inout = radio(pin_id + 'inout',
        {'input': [add_pin_input, remove_pin_input],
         'output': [empty, empty]}
    );
    pin_div.appendChild(inout);

}

function empty () {

}


function add_pin_input() {
    const parent = this.parentElement.parentElement;
    const div1 = document.createElement('div');
    div1.id = this.closest('div[name="pin_div"]') + '_input';
    div1.setAttribute('name', 'pin_input');
    div1.style = "margin-left:2em;"
    parent.appendChild(div1);

    const value_element = text('Value: ', div1.id + '_value');
    div1.appendChild(value_element);

    const req_opt = radio(parent.id + '_req_opt',
        {'required': [empty, empty],
         'optional': [add_pin_optional, remove_pin_optional]}
    );
    div1.appendChild(req_opt);
}

function remove_pin_input() {
    const parent = this.parentElement.parentElement;
    const to_delete = parent.querySelector('div[name="pin_input"]');
    parent.removeChild(to_delete);

}

function add_pin_optional() {
    const parent = this.parentElement.parentElement;
    const div1 = document.createElement('div');
    div1.id = this.closest('div[name="pin_div"]') + '_optional';
    div1.setAttribute('name', 'pin_optional');
    div1.style = "margin-left:2em;"
    parent.appendChild(div1);

    const opt_type = radio(parent.parentElement.id + '_opt_type',
        {'analog': [empty, empty],
         'quantized_analog': [empty, empty],
         'true_digital': [empty, empty]}
    );
    div1.appendChild(opt_type);
}

function remove_pin_optional() {
    const parent = this.parentElement.parentElement;
    const to_delete = parent.querySelector('div[name="pin_optional"]');
    parent.removeChild(to_delete);

}


function label(name, element) {
    const label_element = document.createElement('label');
    label_element.innerHTML = name;
    label_element.for = element;
    const div = document.createElement('div');
    div.appendChild(label_element);
    div.appendChild(element);
    return div;
}

function text(prompt, id) {
    const text_box = document.createElement('input');
    text_box.type = 'text';
    text_box.id = id;
    const text_element = label(prompt, text_box);
    return text_element;
}

function radio(name, choices) {
    const div = document.createElement('div');
    let current_value = null;
    const input_objects = {}
    for (let choice in choices) {

        const input = document.createElement('input');
        input.type = 'radio';
        input.name = name;
        const id = name + '_' + choice;
        input.id = id;
        input_objects[choice] = input;
        div.appendChild(input);

        const label = document.createElement('label');
        label.for = id;
        label.innerHTML = choice;
        div.appendChild(label);

        const span = document.createElement('span');
        span.id = 'radio_spacing';
        span.style = 'width:1em;display:inline-block;';
        div.appendChild(span);
    };

    // implement onselect and ondeselect
    for (let choice in choices) {

        function onclick() {
            if (current_value == choice) {
                return;
            }
            // deselect old
            if (current_value != null) {
                choices[current_value][1].call(input_objects[current_value]);
            }
            // select new
            choices[choice][0].call(input_objects[choice]);
            current_value = choice
        }

        input_objects[choice].onclick = onclick;
    }

    return div;
}

function dropdown(name, choices) {
    const select = document.createElement('select');
    select.name = name;
    select.id = name;

    choices.forEach(choice => {
        var option = document.createElement('option');
        option.value = choice;
        option.text = choice;
        select.appendChild(option);
    });

    return select;
}

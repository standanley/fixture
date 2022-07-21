let pin_name_inc = 0;
let circuit_dict = {}


function add_pin() {
    const pin_container = document.getElementById('pin_container');
    const pin_div = document.createElement('div')
    const pin_id = 'pin' + String(pin_name_inc);
    pin_name_inc++;
    pin_div.id = pin_id;
    pin_div.setAttribute('name', 'pin_div')
    pin_container.appendChild(pin_div);

    pin_div.appendChild(document.createElement('br'));


    const name_element = text('Pin name: ', pin_id + '_name');
    pin_div.appendChild(name_element);

    const electricaltype = radio(pin_id + '_electricaltype',
        {'voltage': [empty, empty],
         'current': [empty, empty]}
    );
    pin_div.appendChild(electricaltype);


    const inout = radio(pin_id + '_inout',
        {'input': [add_pin_input, remove_pin_input],
         'output': [add_pin_output, remove_pin_output]}
    );
    pin_div.appendChild(inout);

}

function empty () {

}

function add_section(input_element, suffix) {
    const parent = input_element.parentElement.parentElement;
    const div1 = document.createElement('div');
    pinid = input_element.closest('div[name="pin_div"]').id
    div1.id = pinid + suffix;
    div1.setAttribute('name', 'pin' + suffix);
    div1.style = "margin-left:2em;"
    parent.appendChild(div1);
    return [div1, pinid];
}

function remove_section(input_element, suffix) {
    const parent = input_element.parentElement.parentElement;
    const to_delete = parent.querySelector('div[name="pin'+suffix+'"]');
    parent.removeChild(to_delete);
}


function add_pin_input() {
    const [div1, pinid] = add_section(this, '_input');

    const value_element = text('Value: ', pinid + '_value');
    div1.appendChild(value_element);

    const req_opt = radio(pinid + '_req_opt',
        {'required': [add_pin_required, remove_pin_required],
         'optional': [add_pin_optional, remove_pin_optional]}
    );
    div1.appendChild(req_opt);
}
function remove_pin_input() {
    remove_section(this, '_input');
}


function add_pin_output() {
    const [div1, pinid] = add_section(this, '_output');
    const out_type = radio(pinid + '_out_type',
        {
            'bit': [empty, empty],
            'real': [empty, empty]
        }
    );
    div1.appendChild(out_type)
}
function remove_pin_output() {
    remove_section(this, '_output');
}


function add_pin_required() {
    const [div1, pinid] = add_section(this, '_required');
    const req_type = radio(pinid + '_req_type',
        {
            'bit': [empty, empty],
            'real': [empty, empty]
        }
    );
    div1.appendChild(req_type)
}
function remove_pin_required() {
    remove_section(this, '_required');
}



function add_pin_optional() {
    const [div1, pinid] = add_section(this, '_optional');

    const opt_type = radio(pinid + '_opt_type',
        {'analog': [empty, empty],
         'quantized_analog': [add_pin_qa, remove_pin_qa],
         'true_digital': [empty, empty]}
    );
    div1.appendChild(opt_type);
}
function remove_pin_optional() {
    remove_section(this, '_optional');
}


function add_pin_qa() {
    const [div1, pinid] = add_section(this, '_qa');

    const qa = radio(pinid + '_qa',
        {'binary': [empty, empty],
         'signed_magnitude': [empty, empty],
         'thermometer': [empty, empty],
         'one_hot': [empty, empty]}
    );
    div1.appendChild(qa);

    const first_one = radio(pinid + '_first_one',
        {'first_one_low': [empty, empty],
         'first_one_high': [empty, empty]}
    );
    div1.appendChild(first_one);
}
function remove_pin_qa() {
    remove_section(this, '_qa');
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
        label.setAttribute('for', id);
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


function finish() {
    const config = [];

    config.push('pins:')
    const pin_container = document.getElementById('pin_container');
    for (var i = 0; i < pin_container.children.length; i++) {
        pin_element = pin_container.children[i];

        const name = pin_element.querySelector('#' + pin_element.id + '_name').value
        config.push('    name: ' + name)

        const electricaltype = get_radio_value(pin_element, '_electricaltype')
        // config considers absence of this tag to mean voltage
        if (electricaltype == 'current') {
            config.push('        electricaltype: current');
        }

        const direction = get_radio_value(pin_element, '_inout');
        config.push('        direction: ' + direction);

        const input_element = pin_element.querySelector('div[name="pin_input"]');
        if (input_element != null) {
            req_opt = get_radio_value(pin_element, '_req_opt');
            if (req_opt == 'required') {
                const req_type = get_radio_value(pin_element, '_req_type');
                config.push('        datatype: ' + req_type);
            } else if (req_opt == 'optional') {
                const opt_type = get_radio_value(pin_element, '_opt_type');
                config.push('        datatype: ' + opt_type);

                if (opt_type == 'quantized_analog') {
                    qa_style = get_radio_value(pin_element, '_qa');
                    first_one_verbose = get_radio_value(pin_element, '_first_one');
                    first_one = first_one_verbose.substring(10);
                    config.push('        bus_style: ' + qa_style);
                    config.push('        first_one: ' + first_one);
                }
            }
        }

        const output_element = pin_element.querySelector('div[name="pin_output"]');
        if (output_element != null) {
            const out_type = get_radio_value(pin_element, '_out_type');
            config.push('        datatype: ' + out_type);
        }


    }

    console.log(config);
    const config_pre = document.getElementById('config');
    config_pre.innerHTML = config.join('\n');
}

function get_radio_value(pin_element, suffix) {
    const radio_elem = pin_element.querySelector('input[name="' + pin_element.id + suffix + '"]:checked');
    if (radio_elem == null) {
        return null;
    }
    value = pin_element.querySelector('label[for="'+radio_elem.id+'"]').innerHTML;
    return value;
}
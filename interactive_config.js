let pinIdInc = 0;
let filepath;
const pinsConfig = {};

/**
 * Handles adding a new pin to the config
 * @returns {void}
 */
function addPin() {
  const pinId = pinIdInc;
  pinsConfig[pinId] = {};

  const pinsContainer = document.getElementById("pins-container");
  const pinDiv = document.createElement("div");
  pinDiv.classList.add("pin");
  pinDiv.id = pinId;

  const electricalType = _createRadioFieldset({
    name: "electricaltype",
    legend: "Electrical type",
    inputs: [
      {
        val: "voltage",
        onchange: "updatePinsConfig",
      },
      {
        val: "current",
        onchange: "updatePinsConfig",
      },
    ],
    pinId,
  });

  const io = _createRadioFieldset({
    name: "direction",
    legend: "I/O",
    inputs: [
      {
        val: "input",
        onchange: "updateIO",
      },
      {
        val: "output",
        onchange: "updateIO",
      },
    ],
    pinId,
  });

  pinDiv.innerHTML = `
    <span>
      <label for="pin-name--${pinId}">Pin name:</label>
      <input type="text" id="pin-name--${pinId}" onchange="updatePinsConfig(${pinId}, 'name', this)"/>
    </span>
    ${electricalType.outerHTML}
    ${io.outerHTML}
    <button class="pin-delete-btn" type="button" onclick="deletePin(this)">Delete pin</button>
  `;

  pinsContainer.appendChild(pinDiv);
  pinIdInc += 1;
  _refreshConfig();
}

/**
 * Deletes a pin from the config
 * @param {Element} parentElement
 * @returns {void}
 */
function deletePin({ parentElement }) {
  parentElement.remove();
  delete pinsConfig[parentElement.id];
  _refreshConfig();
}

/**
 * Validates the config file and downloads if valid
 * @returns {Boolean|void}
 */
function downloadConfig() {
  if (!filepath) {
    alert("Please enter a circuit filepath for your .yaml config file");
    return false;
  }
  if (Object.keys(pinsConfig).length === 0) {
    alert("Your config has no pins");
    return false;
  }
  for (const pinData of Object.values(pinsConfig)) {
    if (!_isValidPin(pinData)) {
      alert("Your config is missing required data for your pins");
      return false;
    }
  }

  const configFilepath = filepath.endsWith(".yaml")
    ? filepath
    : `${filepath}.yaml`;
  const configText = document.getElementById("config-preview").innerHTML;

  const element = document.createElement("a");
  element.setAttribute(
    "href",
    "data:text/plain;charset=utf-8," + encodeURIComponent(configText)
  );
  element.setAttribute("download", configFilepath);
  element.style.display = "none";

  document.body.appendChild(element);
  element.click();
  document.body.removeChild(element);
}

/**
 * Handles a change to the bustypes fieldset
 * @param {Number} pinId
 * @param {String} field
 * @param {Element} elem
 * @returns {void}
 */
function updateBusType(pinId, field, elem) {
  if (elem.value === "signed_magnitude") {
    const signBitLocs = _createRadioFieldset({
      name: "signbit",
      legend: "Sign bit location",
      pinId,
      inputs: [
        {
          val: "low",
          onchange: "updatePinsConfig",
        },
        {
          val: "high",
          onchange: "updatePinsConfig",
        },
      ],
    });

    elem.parentElement.appendChild(signBitLocs);
  } else {
    _removeFieldsets(pinId, ["signbit"]);
  }

  updatePinsConfig(...arguments);
}

/**
 * Updates the filepath of the config file
 * @param {Element.value} value
 * @returns {void}
 */
function updateFilepath({ value }) {
  filepath = value;
  _refreshConfig();
}

/**
 * Handles a change to the input/output fieldset
 * @param {Number} pinId
 * @param {String} field
 * @param {Element} elem
 * @returns {void}
 */
function updateIO(pinId, field, elem) {
  if (elem.value === "input") {
    _removeFieldsets(pinId, ["datatype"]);
    const inputVals = _createInputValsFieldset(pinId);
    const required = _createRequiredFieldset(pinId);
    const datatypes = _createOptionalDatatypesFieldset(pinId);

    elem.parentElement.appendChild(inputVals);
    elem.parentElement.appendChild(required);
    elem.parentElement.appendChild(datatypes);
  } else {
    _removeFieldsets(pinId, [
      "datatype",
      "inputvals",
      "required",
      "bustype",
      "firstone",
      "signbit",
      "min",
      "max",
      "value",
    ]);
    const datatypes = _createRealBitDatatypesFieldset(pinId);
    elem.parentElement.appendChild(datatypes);
  }

  updatePinsConfig(...arguments);
}

/**
 * Handles a change to the optional datatypes fieldset
 * @param {Number} pinId
 * @param {String} field
 * @param {Element} elem
 * @returns {void}
 */
function updateOptionalDatatypes(pinId, field, elem) {
  if (elem.value === "quantized_analog") {
    const busTypes = _createRadioFieldset({
      name: "bustype",
      legend: "Bus type",
      pinId,
      inputs: [
        {
          val: "binary",
          onchange: "updateBusType",
        },
        {
          val: "signed_magnitude",
          onchange: "updateBusType",
        },
        {
          val: "thermometer",
          onchange: "updateBusType",
        },
        {
          val: "one_hot",
          onchange: "updateBusType",
        },
        {
          val: "binary_exact",
          onchange: "updateBusType",
        },
      ],
    });

    const firstOneLocs = _createRadioFieldset({
      name: "firstone",
      legend: "First one location",
      pinId,
      inputs: [
        {
          val: "low",
          onchange: "updatePinsConfig",
        },
        {
          val: "high",
          onchange: "updatePinsConfig",
        },
      ],
    });

    elem.parentElement.parentElement.appendChild(busTypes);
    elem.parentElement.parentElement.appendChild(firstOneLocs);
  } else {
    _removeFieldsets(pinId, ["bustype", "firstone", "signbit"]);
  }

  updatePinsConfig(...arguments);
}

/**
 * Updates the pins config with the new value of the given field
 * @param {Number} pinId
 * @param {String} field
 * @param {Element.value} value
 * @returns {void}
 */
function updatePinsConfig(pinId, field, { value }) {
  pinsConfig[pinId][field] = value;
  _refreshConfig();
}

/**
 * Handles when the required checkbox is toggled
 * @param {Number} pinId
 * @param {Element} elem
 * @returns {void}
 */
function toggleRequired(pinId, elem) {
  if (elem.checked) {
    _removeFieldsets(pinId, ["datatype", "bustype", "firstone", "signbit"]);
    const datatypes = _createRealBitDatatypesFieldset(pinId);
    elem.parentElement.parentElement.appendChild(datatypes);
  } else {
    _removeFieldsets(pinId, ["datatype"]);
    const datatypes = _createOptionalDatatypesFieldset(pinId);
    elem.parentElement.parentElement.appendChild(datatypes);
  }
  _refreshConfig();
}

/**
 * Creates a fieldset for the acceptable input values
 * @param {Number} pinId
 * @returns {Element}
 */
function _createInputValsFieldset(pinId) {
  const name = "inputvals";
  const fieldsetId = `pin-${name}--${pinId}`;
  const fieldsetElem = document.createElement("fieldset");
  fieldsetElem.id = fieldsetId;
  fieldsetElem.classList.add("pin-field", "pin-input-vals");

  const inputsInner = ["min", "max", "value"].map((val) => {
    const inputId = `pin-${name}__${val}--${pinId}`;
    return `
      <label for="${inputId}" class="input-number__label">${val}:</label>
    	<input type="number" id="${inputId}" class="input-number" name="${fieldsetId}" onchange="updatePinsConfig(${pinId}, '${val}', this)"/>  
    `;
  });

  fieldsetElem.innerHTML = `
    <legend>Acceptable input values</legend>
    ${inputsInner.join("")}
  `;
  return fieldsetElem;
}

/**
 * Creates a fieldset for the optional datatypes
 * @param {Number} pinId
 * @returns {Element}
 */
function _createOptionalDatatypesFieldset(pinId) {
  return _createRadioFieldset({
    name: "datatype",
    legend: "Data type",
    pinId,
    inputs: [
      {
        val: "analog",
        onchange: "updateOptionalDatatypes",
      },
      {
        val: "quantized_analog",
        onchange: "updateOptionalDatatypes",
      },
      {
        val: "true_digital",
        onchange: "updateOptionalDatatypes",
      },
    ],
  });
}

/**
 * Creates a fieldset for radio inputs
 * @param {String} options.name name of the fieldset
 * @param {String} options.legend legend information for the fieldset
 * @param {Object[]} options.inputs
 * @param {String} options.inputs[].val value of the radio input
 * @param {String} options.inputs[].onchange name of the function that gets called for the onchange handler
 * @param {Number} options.pinId id of the pin this fieldset is associated with
 * @returns {Element}
 */
function _createRadioFieldset({ name, legend, inputs, pinId }) {
  const fieldsetId = `pin-${name}--${pinId}`;

  const fieldsetElem = document.createElement("fieldset");
  fieldsetElem.id = fieldsetId;
  fieldsetElem.classList.add("pin-field");

  const legendInner = `<legend>${legend}</legend>`;
  const inputsInner = inputs.map(({ val, onchange }) => {
    const inputId = `pin-${name}__${val}--${pinId}`;
    return `
  		<input type="radio" id="${inputId}" name="${fieldsetId}" value="${val}" onchange="${onchange}(${pinId}, '${name}', this)"/>
			<label for="${inputId}">${val}</label>
  	`;
  });

  fieldsetElem.innerHTML = `
    ${legendInner}
    ${inputsInner.join("")}
  `;

  return fieldsetElem;
}

/**
 * Creates a fieldset for the real/bit datatypes
 * @param {Number} pinId
 * @returns {Element}
 */
function _createRealBitDatatypesFieldset(pinId) {
  return _createRadioFieldset({
    name: "datatype",
    legend: "Data type",
    pinId,
    inputs: [
      {
        val: "real",
        onchange: "updatePinsConfig",
      },
      {
        val: "bit",
        onchange: "updatePinsConfig",
      },
    ],
  });
}

/**
 * Creates a fieldset for the required field
 * @param {Number} pinId
 * @returns {Element}
 */
function _createRequiredFieldset(pinId) {
  const name = "required";
  const fieldsetId = `pin-${name}--${pinId}`;
  const fieldsetElem = document.createElement("fieldset");
  fieldsetElem.id = fieldsetId;
  fieldsetElem.classList.add("pin-field");
  fieldsetElem.innerHTML = `
    <legend>Additional options</legend>
    <input type="checkbox" id="pin-${name}__req--${pinId}" onchange="toggleRequired(${pinId}, this)">
    <label for="pin-${name}__req--${pinId}">required</label>
  `;
  return fieldsetElem;
}

/**
 * Validates that the required fields of the pin data are present
 * @param {Object} pinData
 * @returns {Boolean}
 */
function _isValidPin(pinData) {
  const {
    name,
    bustype,
    datatype,
    direction,
    electricaltype,
    firstone,
    max,
    min,
    signbit,
    value,
  } = pinData;

  if (!(name && datatype && direction && electricaltype)) {
    return false;
  }
  if (direction === "input") {
    if (!(min || max || value)) {
      return false;
    }
    if ((min && !max) || (!min && max)) {
      return false;
    }

    if (datatype === "quantized_analog") {
      if (!(bustype && firstone)) {
        return false;
      }

      if (bustype === "signed_magnitude") {
        if (!signbit) {
          return false;
        }
      }
    }
  }
  return true;
}

/**
 * Removes the fields from a specific pin from the UI and config
 * @param {Number} pinId
 * @param {String[]} fields
 * @returns {void}
 */
function _removeFieldsets(pinId, fields) {
  fields.forEach((field) => {
    const elem = document.getElementById(`pin-${field}--${pinId}`);
    if (elem) {
      elem.remove();
    }
    delete pinsConfig[pinId][field];
  });
}

/**
 * Refreshes the config preview
 * @returns {void}
 */
function _refreshConfig() {
  const configLines = [`filepath: ${filepath || ""}`, "pins:"];
  Object.entries(pinsConfig).forEach(([pinId, pinData]) => {
    configLines.push(`    name: ${pinData.name || ""}`);
    Object.entries(pinData).forEach(([key, val]) => {
      if (key !== "name") {
        configLines.push(`        ${key}: ${val || ""}`);
      }
    });
  });

  document.getElementById("config-preview").innerHTML = configLines.join("\n");
}
